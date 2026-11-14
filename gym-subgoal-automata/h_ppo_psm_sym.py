# PSM variant adapted for gym-subgoal-automata (OfficeWorld/WaterWorld).
import math
import random
import time
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append(str(Path(__file__).parent.parent))

from config import Args
import gym
import gym.wrappers
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            params={
                "generation": "random",
                "environment_seed": seed,
                "use_one_hot_vector_states": True,
            },
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.observation_space, spaces.Discrete):
            n = env.observation_space.n
            env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def total_variation_distance_matrix(p1, p2):
    return 0.5 * torch.abs(p1[:, None, :] - p2[None, :, :]).sum(dim=-1)


def build_transition_terminals(dones):
    """Marks whether the transition out of each rollout step is terminal.

    In CleanRL-style storage, `dones[t]` is the done flag *before* stepping from
    `obs[t]`, so transition-terminal flags must be shifted by one step.
    """
    transition_terminals = torch.zeros_like(dones, dtype=torch.bool)
    if dones.shape[0] > 1:
        transition_terminals[:-1] = dones[1:].bool()
    transition_terminals[-1] = True
    return transition_terminals


def compute_policy_similarity_metric_fixed_point(
    policy_distance_matrix,
    has_successor_x,
    has_successor_y,
    gamma,
    dp_eps,
):
    """Solver for the deterministic D.2 recursion, with terminal-transition masking.

    The recursion only looks one step ahead, d(i, j) = D(i, j) + gamma * d(i+1, j+1),
    so its fixed point is reached exactly by one backward pass over the rollout
    steps. This gives the same result as the `while True` fixed-point iteration of
    jumping_task, without its ~num_x full-matrix passes; dp_eps is no longer needed.
    """
    num_x, num_y = policy_distance_matrix.shape
    if num_x == 0 or num_y == 0:
        return policy_distance_matrix

    # Paper D.2 convention: DIST between terminal states is zero.
    terminal_pair_mask = (~has_successor_x)[:, None] & (~has_successor_y)[None, :]
    effective_policy_distance = torch.where(
        terminal_pair_mask,
        torch.zeros_like(policy_distance_matrix),
        policy_distance_matrix,
    )

    current_metric = torch.zeros_like(policy_distance_matrix)
    next_x = (torch.arange(num_x, device=policy_distance_matrix.device) + 1).clamp(max=num_x - 1)
    next_y = (torch.arange(num_y, device=policy_distance_matrix.device) + 1).clamp(max=num_y - 1)
    future_mask = has_successor_x[:, None] & has_successor_y[None, :]
    zero_future = torch.zeros_like(policy_distance_matrix)

    for i in reversed(range(num_x)):
        shifted_future = current_metric[next_x[i]][next_y]
        current_metric[i] = effective_policy_distance[i] + gamma * torch.where(
            future_mask[i],
            shifted_future,
            zero_future[i],
        )
    return current_metric


class LazyPairCache(dict):
    """Dict of env pairs whose values are computed the first time they are requested."""

    def __init__(self, compute_fn):
        super().__init__()
        self.compute_fn = compute_fn

    def __missing__(self, key):
        value = self.compute_fn(key)
        self[key] = value
        return value


def compute_psm_contrastive_loss(similarity_matrix, gamma_matrix, temperature):
    n, m = similarity_matrix.shape
    if n == 0 or m == 0:
        return similarity_matrix.new_zeros(())

    scaled = similarity_matrix / max(temperature, 1e-8)
    pos_x_idx = torch.argmax(gamma_matrix, dim=0)
    col_idx = torch.arange(m, device=similarity_matrix.device)
    pos_logits_1 = scaled[pos_x_idx, col_idx]
    pos_weight_1 = gamma_matrix[pos_x_idx, col_idx].clamp_min(1e-8)
    logits_1 = scaled + torch.log((1.0 - gamma_matrix).clamp_min(1e-8))
    logits_1 = logits_1.clone()
    logits_1[pos_x_idx, col_idx] = pos_logits_1 + torch.log(pos_weight_1)
    loss_1 = (torch.logsumexp(logits_1, dim=0) - (pos_logits_1 + torch.log(pos_weight_1))).mean()

    pos_y_idx = torch.argmax(gamma_matrix, dim=1)
    row_idx = torch.arange(n, device=similarity_matrix.device)
    pos_logits_2 = scaled[row_idx, pos_y_idx]
    pos_weight_2 = gamma_matrix[row_idx, pos_y_idx].clamp_min(1e-8)
    logits_2 = scaled + torch.log((1.0 - gamma_matrix).clamp_min(1e-8))
    logits_2 = logits_2.clone()
    logits_2[row_idx, pos_y_idx] = pos_logits_2 + torch.log(pos_weight_2)
    loss_2 = (torch.logsumexp(logits_2, dim=1) - (pos_logits_2 + torch.log(pos_weight_2))).mean()

    return loss_1 + loss_2


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Same layers and initialisation as the PPO actor, split so that the PSM
        # embedding is computed from the hidden representation used by the policy.
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.projection = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), std=1.0),
        )
        self.conf_level = 0.8
        self.num_actions = envs.single_action_space.n
        self.grounded_dim = 64
        self._office_lookup_cache = {}

        # Extract static environment metadata once so grounded vectors can be
        # computed from one-hot/vector observations only.
        self.env_kind = "unknown"
        self.office_metadata = []
        self.water_metadata = []
        unwrapped_envs = [self._unwrap_env(e) for e in envs.envs]
        env_class_name = unwrapped_envs[0].__class__.__name__ if len(unwrapped_envs) > 0 else ""
        if "OfficeWorld" in env_class_name:
            self.env_kind = "officeworld"
            for env in unwrapped_envs:
                room_positions = {}
                office_pos = None
                plant_positions = []
                for pos, obj in env.locations.items():
                    if obj == "g":
                        office_pos = pos
                    elif obj in ("a", "b", "c", "d"):
                        room_positions[obj] = pos
                    elif obj == "n":
                        plant_positions.append(pos)
                self.office_metadata.append(
                    {
                        "width": int(env.width),
                        "height": int(env.height),
                        "office": office_pos,
                        "mail": tuple(env.mail) if env.mail is not None else None,
                        "coffee": [tuple(p) for p in env.coffee],
                        "plants": [tuple(p) for p in plant_positions],
                        "rooms": room_positions,
                    }
                )
            # Build static grounded-vector tables per env/state for fast one-hot lookup.
            self.office_grounded_lookup = [self._build_office_grounded_lookup(meta) for meta in self.office_metadata]
        elif "WaterWorld" in env_class_name:
            self.env_kind = "waterworld"
            for env in unwrapped_envs:
                self.water_metadata.append(
                    {
                        "num_colors": int(env.ball_num_colors),
                        "num_per_color": int(env.ball_num_per_color),
                        "use_velocities": bool(env.use_velocities),
                    }
                )

    def _unwrap_env(self, env):
        current = env
        while hasattr(current, "env"):
            current = current.env
        if hasattr(current, "unwrapped"):
            return current.unwrapped
        return current

    def get_value(self, x):
        return self.critic(x)

    def get_policy_probs(self, x):
        return torch.softmax(self.actor_head(self.actor_trunk(x)), dim=-1)

    def _decode_office_state_id(self, state_id, width, height):
        possible_values = [width, height, 2, 2, 4]
        content_index = width * height * 2 * 2 * 4
        vars_decoded = []
        remaining = int(state_id)
        for value in possible_values:
            content_index //= value
            var = remaining // content_index
            remaining = remaining % content_index
            vars_decoded.append(int(var))
        # x, y, has_coffee, has_mail, visited_rooms
        return vars_decoded

    def _set_landmark_slot(self, vec_row, start, ax, ay, target, width, height):
        if target is None:
            return
        tx, ty = target
        dx_raw = float(tx - ax)
        dy_raw = float(ty - ay)
        dx = dx_raw / max(1.0, float(width - 1))
        dy = dy_raw / max(1.0, float(height - 1))
        manhattan = abs(dx_raw) + abs(dy_raw)

        vec_row[start] = 1.0
        vec_row[start + 1] = max(dx, 0.0)
        vec_row[start + 2] = max(-dx, 0.0)
        vec_row[start + 3] = max(dy, 0.0)
        vec_row[start + 4] = max(-dy, 0.0)
        vec_row[start + 5] = manhattan / max(1.0, float((width - 1) + (height - 1)))
        vec_row[start + 6] = 1.0 if (ax, ay) == (tx, ty) else 0.0
        vec_row[start + 7] = 1.0 if manhattan == 1.0 else 0.0

    def _nearest_position(self, origin, positions):
        if positions is None or len(positions) == 0:
            return None
        ox, oy = origin
        return min(positions, key=lambda p: abs(p[0] - ox) + abs(p[1] - oy))

    def _build_office_grounded_lookup(self, meta):
        width = int(meta["width"])
        height = int(meta["height"])
        num_states = width * height * 2 * 2 * 4
        lookup = torch.zeros((num_states, self.grounded_dim), dtype=torch.float32)

        room_positions = meta["rooms"]
        for state_id in range(num_states):
            ax, ay, has_coffee, has_mail, visited = self._decode_office_state_id(state_id, width, height)
            row = lookup[state_id]
            row[0] = float(ax) / max(1.0, float(width - 1))
            row[1] = float(ay) / max(1.0, float(height - 1))
            row[2] = float(has_coffee)
            row[3] = float(has_mail)
            row[4 + int(visited)] = 1.0

            current_pos = (ax, ay)
            nearest_coffee = self._nearest_position(current_pos, meta["coffee"])
            nearest_plant = self._nearest_position(current_pos, meta["plants"])
            landmarks = [
                meta["office"],
                meta["mail"],
                nearest_coffee,
                nearest_plant,
                room_positions.get("a"),
                room_positions.get("b"),
                room_positions.get("c"),
            ]
            for landmark_idx, landmark in enumerate(landmarks):
                start = 8 + 8 * landmark_idx
                self._set_landmark_slot(row, start, ax, ay, landmark, width, height)

        return lookup

    def _get_office_lookup_on_device(self, env_idx, x):
        key = (env_idx, str(x.device), str(x.dtype))
        table = self._office_lookup_cache.get(key)
        if table is None:
            table = self.office_grounded_lookup[env_idx].to(device=x.device, dtype=x.dtype)
            self._office_lookup_cache[key] = table
        return table

    def _get_office_grounded_vectors(self, x, env_indices):
        if isinstance(env_indices, int):
            lookup = self._get_office_lookup_on_device(env_indices, x)
            return x @ lookup

        vectors = torch.zeros((x.shape[0], self.grounded_dim), device=x.device, dtype=x.dtype)
        env_tensor = torch.as_tensor(env_indices, device=x.device, dtype=torch.long)
        for env_idx in torch.unique(env_tensor).tolist():
            mask = env_tensor == int(env_idx)
            lookup = self._get_office_lookup_on_device(int(env_idx), x)
            vectors[mask] = x[mask] @ lookup
        return vectors

    def _get_water_grounded_vectors(self, x, env_indices):
        if isinstance(env_indices, int):
            env_indices = [env_indices] * x.shape[0]

        vectors = torch.zeros((x.shape[0], self.grounded_dim), device=x.device, dtype=x.dtype)
        env_tensor = torch.as_tensor(env_indices, device=x.device, dtype=torch.long)

        for env_idx in torch.unique(env_tensor).tolist():
            mask = env_tensor == int(env_idx)
            if torch.sum(mask).item() == 0:
                continue

            x_env = x[mask]
            meta = self.water_metadata[int(env_idx)]
            num_colors = meta["num_colors"]
            num_per_color = meta["num_per_color"]
            use_velocities = meta["use_velocities"]
            per_ball_stride = 4 if use_velocities else 2

            vec_env = torch.zeros((x_env.shape[0], self.grounded_dim), device=x.device, dtype=x.dtype)
            vec_env[:, 0:4] = x_env[:, 0:4]

            balls = x_env[:, 4:].reshape(x_env.shape[0], num_colors, num_per_color, per_ball_stride)
            rel_pos = balls[:, :, :, 0:2]
            dists = torch.norm(rel_pos, dim=3)
            nearest_idx = torch.argmin(dists, dim=2)

            gather_pos_idx = nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            nearest_pos = rel_pos.gather(2, gather_pos_idx).squeeze(2)
            min_dist = dists.gather(2, nearest_idx.unsqueeze(-1)).squeeze(-1)
            mean_dist = dists.mean(dim=2)

            if use_velocities:
                rel_vel = balls[:, :, :, 2:4]
                gather_vel_idx = nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
                nearest_vel = rel_vel.gather(2, gather_vel_idx).squeeze(2)
                rel_speed = torch.norm(nearest_vel, dim=2)
                denom = torch.clamp(min_dist, min=1e-6)
                approach_score = torch.clamp(-torch.sum(nearest_pos * nearest_vel, dim=2) / denom, min=0.0)
            else:
                rel_speed = torch.zeros_like(min_dist)
                approach_score = torch.zeros_like(min_dist)

            for c in range(num_colors):
                slot = 4 + c * 10
                vec_env[:, slot] = 1.0
                vec_env[:, slot + 1] = torch.clamp(min_dist[:, c], 0.0, 1.0)
                vec_env[:, slot + 2] = torch.clamp(mean_dist[:, c], 0.0, 1.0)
                vec_env[:, slot + 3] = torch.clamp(nearest_pos[:, c, 0], min=0.0)
                vec_env[:, slot + 4] = torch.clamp(-nearest_pos[:, c, 0], min=0.0)
                vec_env[:, slot + 5] = torch.clamp(nearest_pos[:, c, 1], min=0.0)
                vec_env[:, slot + 6] = torch.clamp(-nearest_pos[:, c, 1], min=0.0)
                vec_env[:, slot + 7] = torch.clamp(rel_speed[:, c], 0.0, 2.0) / 2.0
                vec_env[:, slot + 8] = torch.clamp(approach_score[:, c], 0.0, 2.0) / 2.0
                vec_env[:, slot + 9] = (min_dist[:, c] < 0.08).to(x.dtype)

            vectors[mask] = vec_env

        return vectors

    def get_grounded_vectors(self, x, env_indices):
        if isinstance(env_indices, int):
            env_indices = [env_indices] * x.shape[0]
        elif torch.is_tensor(env_indices):
            env_indices = env_indices.detach().cpu().tolist()

        if self.env_kind == "officeworld":
            return self._get_office_grounded_vectors(x, env_indices)
        if self.env_kind == "waterworld":
            return self._get_water_grounded_vectors(x, env_indices)

        return torch.zeros((x.shape[0], self.grounded_dim), device=x.device, dtype=x.dtype)

    def get_embedding(self, x):
        """PSM embedding: normalised projection of the policy's hidden representation.

        Gradients of the PSM loss flow into actor_trunk, i.e. into the policy itself
        (Agarwal et al., 2021, policy similarity embeddings).
        """
        z = self.projection(self.actor_trunk(x))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    def get_symbolic_probs(self, x, envs):
        sym_probs = torch.ones((x.shape[0], self.num_actions), device=x.device) * (1.0 - self.conf_level)
        for i in range(x.shape[0]):
            for suggested_action in envs.envs[i].guide_agent():
                if suggested_action is not None:
                    sym_probs[i, suggested_action] = self.conf_level
        return sym_probs / sym_probs.sum(dim=-1, keepdim=True)

    def actor(self, x):
        """Policy logits, as the `actor` module of the other agents (used by the evaluation scripts)."""
        return self.actor_head(self.actor_trunk(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor_head(self.actor_trunk(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.task}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"h_ppo_psm_sym_{args.group_name}",
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Separate generator for the PSM env pairs, so that it does not shift the other random streams.
    psm_pair_rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    symbolic_probs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), device=device)
    neural_probs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    envs.seed(args.seed)
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episodes_returns = []
    episodes_lengths = []
    len_ep_ret = 0

    for iteration in trange(1, args.num_iterations + 1, colour="green"):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                symbolic_probs[step] = agent.get_symbolic_probs(next_obs, envs)
                neural_probs[step] = agent.get_policy_probs(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    writer.add_scalar("episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("episodic_length", item["episode"]["l"], global_step)
                    episodes_returns.append(item["episode"]["r"])
                    episodes_lengths.append(item["episode"]["l"])
            if iteration % max(1, math.ceil(args.num_iterations / 10)) == 0 or iteration == args.num_iterations:
                old_len_ep_ret = len_ep_ret
                len_ep_ret = len(episodes_returns)
                num_eps = len_ep_ret - old_len_ep_ret
                if num_eps > 0:
                    mean_episodic_return = np.mean(episodes_returns[-num_eps:])
                    mean_episodic_length = np.mean(episodes_lengths[-num_eps:])
                    tot_mean_ret = np.mean(episodes_returns)
                    tot_mean_len = np.mean(episodes_lengths)
                    tqdm.write(
                        f"global_step={global_step}, mean_episodic_return={mean_episodic_return}, "
                        f"mean_episodic_length={mean_episodic_length}, total_mean_return={tot_mean_ret}, "
                        f"total_mean_length={tot_mean_len}"
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Precompute rollout-level pairwise couplings once per iteration.
        # Symbolic policy on both sides: two states are close if the rules prescribe
        # similar behaviour in them and along the following steps (PSM recursion).
        # Each minibatch uses a single pair, so a pair is computed the first time it is
        # used and cached for the rest of the rollout.
        with torch.no_grad():
            transition_terminals = build_transition_terminals(dones)
            has_successor = torch.arange(args.num_steps, device=device) < (args.num_steps - 1)

        def compute_pair_metric(pair):
            src_env, tgt_env = pair
            with torch.no_grad():
                sym_src = symbolic_probs[:, src_env, :]
                active_src = has_successor & (~transition_terminals[:, src_env])
                sym_tgt = symbolic_probs[:, tgt_env, :]
                active_tgt = has_successor & (~transition_terminals[:, tgt_env])
                dist = total_variation_distance_matrix(sym_src, sym_tgt)
                return compute_policy_similarity_metric_fixed_point(
                    policy_distance_matrix=dist,
                    has_successor_x=active_src,
                    has_successor_y=active_tgt,
                    gamma=args.gamma,
                    dp_eps=args.psm_dp_eps,
                )

        pair_metric_matrix = LazyPairCache(compute_pair_metric)
        pair_gamma = LazyPairCache(lambda pair: torch.exp(-pair_metric_matrix[pair] / max(args.psm_beta, 1e-8)))
        pair_metric = LazyPairCache(lambda pair: pair_metric_matrix[pair].mean())

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        psm_losses = []
        psm_metrics = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # One ordered pair of distinct envs per minibatch, uniformly at random.
                if args.num_envs > 1:
                    src_env, tgt_env = psm_pair_rng.choice(args.num_envs, size=2, replace=False).tolist()
                    pair = (src_env, tgt_env)

                    src_idx = torch.arange(args.num_steps, device=device) * args.num_envs + src_env
                    tgt_idx = torch.arange(args.num_steps, device=device) * args.num_envs + tgt_env
                    src_emb = agent.get_embedding(b_obs[src_idx])
                    tgt_emb = agent.get_embedding(b_obs[tgt_idx])
                    similarity = src_emb @ tgt_emb.T

                    psm_loss = compute_psm_contrastive_loss(
                        similarity_matrix=similarity,
                        gamma_matrix=pair_gamma[pair],
                        temperature=args.psm_temperature,
                    )
                    psm_metric = pair_metric[pair]
                else:
                    psm_loss = b_obs.new_zeros(())
                    psm_metric = b_obs.new_zeros(())

                psm_losses.append(float(psm_loss.item()))
                psm_metrics.append(float(psm_metric.item()))

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.psm_aux_coef * psm_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/psm_loss", np.mean(psm_losses) if psm_losses else 0.0, global_step)
        writer.add_scalar("losses/psm_metric", np.mean(psm_metrics) if psm_metrics else 0.0, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        os.makedirs(f"models/h_ppo_psm_sym_{args.task}{args.run_code}", exist_ok=True)
        model_path = f"models/h_ppo_psm_sym_{args.task}{args.run_code}/h_ppo_psm_sym_seed={args.seed}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
