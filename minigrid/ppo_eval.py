from typing import Callable
import tyro
import time
import gymnasium as gym
import wandb
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config_eval import Args

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    n_keys: int,
    eval_episodes: int,
    run_name: str,
    seed: int,
    group_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    track: bool = False,
    from_config: bool = True,
    random_color: bool = True,
    return_episode_metrics: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    envs = gym.vector.SyncVectorEnv([make_env(env_id, n_keys, 0, capture_video, run_name, random_color)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    if from_config is True:
        args = tyro.cli(Args)
        env_id = args.env_id
        model_path = args.model_path
        eval_episodes = args.eval_episodes
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
        group_name = args.group_name
        capture_video = args.capture_video
        seed = args.seed
        track = args.track
        n_keys = args.n_keys
        wandb_project_name = args.wandb_project_name
        wandb_entity = args.wandb_entity

    if track:
        run_config = vars(args) if from_config else {}

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=run_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=group_name
        )

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in run_config.items()])),
        )

    seeds = random.sample(range(0, 10**9), eval_episodes)
    obs, _ = envs.reset(seed=seeds[0])
    episodic_returns = []
    episodic_actions_mean = []
    episodic_actions_std = []
    episodic_logprobs_mean = []
    episodic_logprobs_std = []
    episodic_entropies_mean = []
    episodic_entropies_std = []
    episodic_values_mean = []
    episodic_values_std = []
    per_step_metrics = []
    episode_actions = []
    episode_logprobs = []
    episode_entropies = []
    episode_values = []
    episode_probs = []
    total_eval_steps = 0
    pbar = tqdm(total=eval_episodes)
    tqdm.write(f"Model path: {model_path}")
    tqdm.write(f"Evaluating {env_id} with {n_keys} keys, {eval_episodes} episodes")

    while len(episodic_returns) < eval_episodes:
        obs_t = torch.Tensor(obs).to(device)
        actions, logprobs, entropies, values = agent.get_action_and_value(obs_t)
        probs = torch.softmax(agent.actor(obs_t), dim=-1)
        action_val = float(actions.item())
        logprob_val = float(logprobs.item())
        entropy_val = float(entropies.item())
        value_val = float(values.item())
        probs_val = probs.cpu().detach().numpy().tolist()[0]

        episode_actions.append(action_val)
        episode_logprobs.append(logprob_val)
        episode_entropies.append(entropy_val)
        episode_values.append(value_val)
        episode_probs.append(probs_val)

        if return_episode_metrics:  # only kept when the caller asks for them (saves ~0.7 kB per step)
            per_step_metrics.append(
                {
                    "Episode": len(episodic_returns) + 1,
                    "Step": len(episode_actions),
                    "Action": action_val,
                    "LogProb": logprob_val,
                    "Entropy": entropy_val,
                    "Value": value_val,
                    "Probs": probs_val
                }
            )
        total_eval_steps += 1
        if track:
            writer.add_scalar("eval_step/action", action_val, total_eval_steps)
            writer.add_scalar("eval_step/logprob", logprob_val, total_eval_steps)
            writer.add_scalar("eval_step/entropy", entropy_val, total_eval_steps)
            writer.add_scalar("eval_step/value", value_val, total_eval_steps)
            writer.add_scalar("eval_step/episode", len(episodic_returns) + 1, total_eval_steps)
            writer.add_scalar("eval_step/episode_step", len(episode_actions), total_eval_steps)

        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episode_action_mean = float(np.mean(episode_actions))
                episode_action_std = float(np.std(episode_actions))
                episode_logprob_mean = float(np.mean(episode_logprobs))
                episode_logprob_std = float(np.std(episode_logprobs))
                episode_entropy_mean = float(np.mean(episode_entropies))
                episode_entropy_std = float(np.std(episode_entropies))
                episode_value_mean = float(np.mean(episode_values))
                episode_value_std = float(np.std(episode_values))
                
                episodic_returns += [info["episode"]["r"]]
                episodic_actions_mean.append(episode_action_mean)
                episodic_actions_std.append(episode_action_std)
                episodic_logprobs_mean.append(episode_logprob_mean)
                episodic_logprobs_std.append(episode_logprob_std)
                episodic_entropies_mean.append(episode_entropy_mean)
                episodic_entropies_std.append(episode_entropy_std)
                episodic_values_mean.append(episode_value_mean)
                episodic_values_std.append(episode_value_std)
                pbar.update(1)
                if len(episodic_returns) % 100 == 0:
                    tqdm.write(f"eval_episode={len(episodic_returns)}, mean_episodic_return={np.mean(episodic_returns):.2f}")
                if track:
                    writer.add_scalar("eval/episodic_action_mean", episode_action_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_action_std", episode_action_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_logprob_mean", episode_logprob_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_logprob_std", episode_logprob_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_entropy_mean", episode_entropy_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_entropy_std", episode_entropy_std, len(episodic_returns))
                    writer.add_scalar("eval/episodic_value_mean", episode_value_mean, len(episodic_returns))
                    writer.add_scalar("eval/episodic_value_std", episode_value_std, len(episodic_returns))

                episode_actions = []
                episode_logprobs = []
                episode_entropies = []
                episode_values = []
                
                if len(episodic_returns) < eval_episodes:
                    next_obs, _ = envs.reset(seed=seeds[len(episodic_returns)])
        obs = next_obs

    if track:
        writer.close()

    if return_episode_metrics:
        return episodic_returns, {
            "per_step": per_step_metrics,
            "per_episode": {
                "action_mean": episodic_actions_mean,
                "action_std": episodic_actions_std,
                "logprob_mean": episodic_logprobs_mean,
                "logprob_std": episodic_logprobs_std,
                "entropy_mean": episodic_entropies_mean,
                "entropy_std": episodic_entropies_std,
                "value_mean": episodic_values_mean,
                "value_std": episodic_values_std,
            },
        }

    return episodic_returns


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from ppo import make_env, Agent
    args = tyro.cli(Args)
    env_id = args.env_id
    n_keys = args.n_keys
    model_path = args.model_path
    eval_episodes = args.eval_episodes
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval"
    capture_video = args.capture_video
    seed = args.seed
    random_color = args.random_color

    # model_path = hf_hub_download(
    #     repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    # )
    evaluate(
        model_path,
        make_env,
        env_id,
        n_keys=n_keys,
        eval_episodes=eval_episodes,
        run_name=run_name,
        seed=seed,
        group_name=args.group_name,
        Model=Agent,
        device="cpu",
        capture_video=capture_video,
        track=args.track,
        random_color=random_color
    )