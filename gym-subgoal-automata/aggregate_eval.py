import numpy as np
import torch
import tyro
import os
import csv
import time
from config_eval import Args

# Import the specific scripts to get their evaluate functions
import ppo_eval
import h_ppo_eval
import ppo_eval_random_rules
import ppo_eval_check_values
from ppo import make_env, Agent as PPOAgent
from ppo_reward_machine import Agent as PPORMAgent
from h_ppo_psm_sym import Agent as HPPOPSMSymAgent
from h_ppo_product import Agent as HPPOProductAgent
from h_ppo_symloss import Agent as HPPOSymLossAgent
from h_ppo_symloss_eps import Agent as HPPOSymLossEpsAgent
import gym


class RuleStatsMonitor(gym.Wrapper):
    """Counts, for every episode, the steps where the rules suggest at least one action
    and the steps where the executed action is one of the suggested ones.

    The suggestions are queried with guide_agent() on the state *before* the step,
    which is deterministic and does not consume random numbers, so the evaluation
    itself is unchanged.
    """

    def __init__(self, env):
        super().__init__(env)
        self.episodes = []  # one dict per completed episode
        self._reset_counters()

    def _reset_counters(self):
        self._steps = 0
        self._rule_steps = 0
        self._rule_matches = 0

    def reset(self, **kwargs):
        self._reset_counters()
        return self.env.reset(**kwargs)

    def step(self, action):
        suggested = [a for a in self.env.unwrapped.guide_agent() if a is not None]
        executed = int(np.asarray(action).item())
        self._steps += 1
        if suggested:
            self._rule_steps += 1
            self._rule_matches += int(executed in suggested)
        obs, reward, done, info = self.env.step(action)
        if done:
            self.episodes.append({"Steps": self._steps, "RuleSteps": self._rule_steps,
                                  "RuleMatches": self._rule_matches})
        return obs, reward, done, info


def make_monitored_env_factory(created_envs):
    """Same signature as ppo.make_env; wraps the env with RuleStatsMonitor and records it."""
    def _make_env(env_id, seed, idx, capture_video, run_name):
        thunk = make_env(env_id, seed, idx, capture_video, run_name)

        def _thunk():
            env = RuleStatsMonitor(thunk())
            created_envs.append(env)
            return env
        return _thunk
    return _make_env


def get_layout_seeds(args, master_seed, run_index):
    """Environment seeds (map layouts) evaluated for one checkpoint."""
    if args.layout_mode == "train":
        return [master_seed + k for k in range(args.num_layouts)]
    if args.layout_mode == "unseen":
        return [args.unseen_seed_offset + 1000 * run_index + k for k in range(args.num_layouts)]
    raise ValueError(f"Unknown layout_mode: {args.layout_mode}, must be 'train' or 'unseen'.")

def get_agent_class(model_name: str):
    """Return the appropriate Agent class based on model name."""
    m_lower = model_name.lower()
    if m_lower == "ppo_reward_machine" or m_lower == "ppo_rm":
        return PPORMAgent
    elif m_lower == "h_ppo_psm_sym":
        return HPPOPSMSymAgent
    elif m_lower == "h_ppo_product":
        return HPPOProductAgent
    elif m_lower == "h_ppo_symloss":
        return HPPOSymLossAgent
    elif m_lower == "h_ppo_symloss_eps":
        return HPPOSymLossEpsAgent
    else:
        return PPOAgent



def _normalize_eval_episode_suffix(model_dir_name: str, eval_episodes: int) -> str:
    """Return model filename stem with canonical _<N>_eval_eps suffix."""
    suffix = f"_{int(eval_episodes)}_eval_eps"
    if model_dir_name.endswith(suffix):
        return model_dir_name

    # Handle legacy/manual suffixes like *_20_eps.
    legacy_suffix = f"_{int(eval_episodes)}_eps"
    if model_dir_name.endswith(legacy_suffix):
        model_dir_name = model_dir_name[: -len(legacy_suffix)]

    return f"{model_dir_name}{suffix}"

def _append_eval_code_suffix(name_stem: str, eval_code: str) -> str:
    """Append eval_code to filename stem when provided and not already present."""
    if not eval_code:
        return name_stem
    if name_stem.endswith(eval_code):
        return name_stem
    return f"{name_stem}{eval_code}"


def _safe_float(value, default=None):
    """Best-effort float conversion; return default on missing/invalid input."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    """Best-effort int conversion; return default on missing/invalid input."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_prob(probs, idx):
    """Safely extract probability idx from a probs-like sequence."""
    if probs is None:
        return None
    try:
        if len(probs) <= idx:
            return None
        return _safe_float(probs[idx], default=None)
    except (TypeError, ValueError, IndexError):
        return None

def main():
    args = tyro.cli(Args)
    
    # 5 different master seeds for the 5 evaluation runs
    master_seeds = args.master_seeds
    print(f"Using Master Seeds: {master_seeds}")
    all_returns = []
    all_step_rows = []
    all_episode_summary_rows = []
    per_seed_step_rows = {}
    per_seed_episode_summary_rows = {}
    
    # Decide which evaluation function to use based on exp_name or other logic
    if "random_rules" in args.eval_type:
        eval_func = ppo_eval_random_rules.evaluate
        eval_type_subfolder = "random_rules"
    elif "check_values" in args.eval_type:
        eval_func = ppo_eval_check_values.evaluate
        eval_type_subfolder = "check_values"
    elif "h_ppo" in args.eval_type or args.epsilon is not None:
        eval_func = h_ppo_eval.evaluate
        eval_type_subfolder = f"h_ppo_eps_{args.epsilon}"
    elif "standard" in args.eval_type:
        eval_func = ppo_eval.evaluate
        eval_type_subfolder = "standard"
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}, must be one of 'standard', 'random_rules', 'check_values', or 'h_ppo'.")
    
    # Print evaluation info
    eval_info = f"eval_type={args.eval_type}"
    if args.epsilon is not None:
        eval_info += f", epsilon={args.epsilon}"
    print(f"Starting aggregate evaluation on {args.env_id} ({eval_info})")
    print(f"{len(master_seeds)} seeds, {args.eval_episodes} episodes each. Total: {len(master_seeds) * args.eval_episodes} episodes.")

    # For full evaluations, keep return-only flow and skip detailed per-step metrics.
    low_eval = args.eval_episodes < 1000

    collect_episode_metrics = low_eval and args.model_name == "ppo" and eval_func in {
        ppo_eval.evaluate,
        ppo_eval_check_values.evaluate,
    }

    # Multi-layout evaluation and/or per-episode rule statistics (activation rate).
    extended = args.num_layouts > 1 or args.layout_mode != "train" or args.log_rule_stats
    extended_rows = []
    if extended:
        assert args.eval_episodes % args.num_layouts == 0, "eval_episodes must be divisible by num_layouts"
        collect_episode_metrics = False
        print(f"Layouts per seed: {args.num_layouts} ({args.layout_mode}), "
              f"{args.eval_episodes // args.num_layouts} episodes each; rule stats: {args.log_rule_stats}")

    for i, s in enumerate(master_seeds):
        print(f"\n--- Run {i+1}/{len(master_seeds)} with Master Seed {s} ---")
        args.seed = s  # Update seed for each run
        
        eval_kwargs = {
            "model_path": args.model_path,
            "make_env": make_env,
            "env_id": args.env_id,
            "eval_episodes": args.eval_episodes,
            "run_name": f"{args.env_id}_seed_{s}",
            "seed": s,
            "group_name": args.group_name,
            "Model": get_agent_class(args.model_name),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "capture_video": False,
            "track": False, # Don't log individual runs
            "from_config": False,
        }

        if eval_func == h_ppo_eval.evaluate:
            eval_kwargs["epsilon"] = args.epsilon

        if eval_func == ppo_eval_check_values.evaluate:
            eval_kwargs["rolling_window_size"] = args.rolling_window_size
            eval_kwargs["value_skip_threshold"] = args.value_skip_threshold

        if extended:
            # One evaluate() call per layout: the layout is fixed by the environment seed,
            # which ppo.make_env passes to the environment as "environment_seed".
            returns = []
            for layout_seed in get_layout_seeds(args, s, i):
                created_envs = []
                layout_kwargs = dict(
                    eval_kwargs,
                    make_env=make_monitored_env_factory(created_envs),
                    eval_episodes=args.eval_episodes // args.num_layouts,
                    seed=layout_seed,
                    run_name=f"{args.env_id}_seed_{s}_layout_{layout_seed}",
                )
                layout_returns = eval_func(**layout_kwargs)
                episodes = created_envs[0].episodes[: len(layout_returns)]
                assert len(episodes) == len(layout_returns), (len(episodes), len(layout_returns))
                for ep_idx, (ret, ep) in enumerate(zip(layout_returns, episodes), start=1):
                    extended_rows.append({
                        "MasterSeed": s,
                        "LayoutSeed": layout_seed,
                        "Episode": ep_idx,
                        "Return": float(ret.item() if hasattr(ret, "item") else ret),
                        **ep,
                    })
                returns.extend(layout_returns)
            metrics = None
        elif collect_episode_metrics:
            returns, metrics = eval_func(**eval_kwargs, return_episode_metrics=True)
        else:
            returns = eval_func(**eval_kwargs)
            metrics = None
        all_returns.extend(returns)

        if metrics is not None:
            per_step = metrics.get("per_step", [])
            if s not in per_seed_step_rows:
                per_seed_step_rows[s] = []
            for row in per_step:
                ep_idx = _safe_int(row.get("Episode"))
                if ep_idx is None or ep_idx < 1 or ep_idx > len(returns):
                    continue
                episode_return = returns[ep_idx - 1]
                probs = row.get("Probs")
                step_row = {
                    "GlobalStep": len(all_step_rows) + 1,
                    "MasterSeed": s,
                    "Episode": ep_idx,
                    "EpisodeStep": _safe_int(row.get("Step")),
                    "Return": float(episode_return.item() if hasattr(episode_return, "item") else episode_return),
                    "Action": _safe_int(row.get("Action")),
                    "LogProb": _safe_float(row.get("LogProb")),
                    "Entropy": _safe_float(row.get("Entropy")),
                    "Value": _safe_float(row.get("Value")),
                    "Prob0": _safe_prob(probs, 0),
                    "Prob1": _safe_prob(probs, 1),
                    "Prob2": _safe_prob(probs, 2),
                    "Prob3": _safe_prob(probs, 3),
                    "Prob4": _safe_prob(probs, 4),
                    "Prob5": _safe_prob(probs, 5),
                    "Prob6": _safe_prob(probs, 6),
                }
                all_step_rows.append(step_row)
                per_seed_step_rows[s].append(step_row)

            per_episode = metrics.get("per_episode", {})
            if s not in per_seed_episode_summary_rows:
                per_seed_episode_summary_rows[s] = []
            for ep_idx, ret in enumerate(returns, start=1):
                episode_row = {
                    "GlobalEpisode": i * args.eval_episodes + ep_idx,
                    "MasterSeed": s,
                    "Episode": ep_idx,
                    "Return": float(ret.item() if hasattr(ret, "item") else ret),
                    "ActionMean": _safe_float(per_episode.get("action_mean", [])[ep_idx - 1] if len(per_episode.get("action_mean", [])) > ep_idx - 1 else None),
                    "ActionStd": _safe_float(per_episode.get("action_std", [])[ep_idx - 1] if len(per_episode.get("action_std", [])) > ep_idx - 1 else None),
                    "LogProbMean": _safe_float(per_episode.get("logprob_mean", [])[ep_idx - 1] if len(per_episode.get("logprob_mean", [])) > ep_idx - 1 else None),
                    "LogProbStd": _safe_float(per_episode.get("logprob_std", [])[ep_idx - 1] if len(per_episode.get("logprob_std", [])) > ep_idx - 1 else None),
                    "EntropyMean": _safe_float(per_episode.get("entropy_mean", [])[ep_idx - 1] if len(per_episode.get("entropy_mean", [])) > ep_idx - 1 else None),
                    "EntropyStd": _safe_float(per_episode.get("entropy_std", [])[ep_idx - 1] if len(per_episode.get("entropy_std", [])) > ep_idx - 1 else None),
                    "ValueMean": _safe_float(per_episode.get("value_mean", [])[ep_idx - 1] if len(per_episode.get("value_mean", [])) > ep_idx - 1 else None),
                    "ValueStd": _safe_float(per_episode.get("value_std", [])[ep_idx - 1] if len(per_episode.get("value_std", [])) > ep_idx - 1 else None),
                }
                all_episode_summary_rows.append(episode_row)
                per_seed_episode_summary_rows[s].append(episode_row)

    # Calculate statistics
    all_returns_arr = np.array(all_returns)
    mean_return = float(np.mean(all_returns_arr)) if len(all_returns_arr) > 0 else 0.0
    std_return = float(np.std(all_returns_arr)) if len(all_returns_arr) > 0 else 0.0
    
    print("\nFinal Results:")
    print(f"Mean Return: {mean_return:.2f}")
    print(f"Std Return: {std_return:.2f}")

    low_eval = args.eval_episodes < 1000

    base_dir = "aggregate_evals_logs" if low_eval else "aggregate_evals"
    if extended:
        base_dir = "aggregate_evals_multilayout"
    if args.output_root:
        base_dir = args.output_root
    target_dir = os.path.join(base_dir, args.env_id.replace(":", "_"), eval_type_subfolder)
    os.makedirs(target_dir, exist_ok=True)
    # It should use model_dir_name = ppo_RM_DeliverCoffee_v1 or similar.
    # We can infer it from the model_path.
    model_dir_name = os.path.basename(os.path.dirname(args.model_path))

    model_file_stem = _normalize_eval_episode_suffix(model_dir_name, args.eval_episodes)
    # Append check_values with its config params so different window/threshold runs are distinct.
    if eval_type_subfolder == "check_values":
        cv_suffix = f"_check_values_w{args.rolling_window_size}_t{args.value_skip_threshold}"
        if not model_file_stem.endswith(cv_suffix):
            model_file_stem = f"{model_file_stem}{cv_suffix}"
            
    if extended:
        model_file_stem = f"{model_file_stem}_{args.layout_mode}_{args.num_layouts}layouts"
    model_file_stem = _append_eval_code_suffix(model_file_stem, args.eval_code)
    csv_path = os.path.join(target_dir, f"{model_file_stem}.csv")

    print(f"Saving returns to {csv_path}")
    if extended:
        # One row per episode, with the layout and the rule statistics.
        fieldnames = ["MasterSeed", "LayoutSeed", "Episode", "Return", "Steps", "RuleSteps", "RuleMatches"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extended_rows)
    elif low_eval:
        # Detailed per-step schema for low-eval runs (< 1000 episodes).
        fieldnames = ["GlobalStep", "MasterSeed", "Episode", "EpisodeStep",
                      "Return", "Action", "LogProb", "Entropy", "Value",
                      "Prob0", "Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_step_rows)
            
        summary_fieldnames = ["GlobalEpisode", "MasterSeed", "Episode", "Return",
                              "ActionMean", "ActionStd", "LogProbMean", "LogProbStd",
                              "EntropyMean", "EntropyStd", "ValueMean", "ValueStd"]
        summary_path = os.path.join(target_dir, f"{model_file_stem}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_summary_rows)
    else:
        # Lean Return-only schema for full-eval runs (>= 1000 episodes).
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Return"])
            for r in all_returns:
                val = r.item() if hasattr(r, "item") else float(r)
                writer.writerow([val])
                
if __name__ == "__main__":
    main()
