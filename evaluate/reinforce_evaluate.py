#!/usr/bin/env python3
"""
Evaluate the best REINFORCE model (prefer best_model.pt), save the mean reward to a file,
and save per-episode results as CSV.

Usage:
    python reinforce_eval_best.py --eval-episodes 20 --seeds 3 --out-file best_model_mean.txt --out-csv reinforce_eval_results.csv
"""
import os
import sys
import argparse
import importlib
import glob
import json
import random
import csv
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_env_class(env_path: str):
    if ":" in env_path:
        module_path, cls_name = env_path.split(":")
    else:
        parts = env_path.split(".")
        module_path = ".".join(parts[:-1])
        cls_name = parts[-1]
    module = importlib.import_module(module_path)
    EnvClass = getattr(module, cls_name)
    return EnvClass

def find_reinforce_model(model_path_hint: str) -> str:
    if model_path_hint:
        p = Path(model_path_hint)
        if p.exists():
            return str(p)

    base = Path("../models/reinforce_engagement")
    candidates = []
    if base.exists():
        candidates.extend(glob.glob(str(base / "**/best_model*.pt"), recursive=True))
        candidates.extend(glob.glob(str(base / "**/best*.pt"), recursive=True))
        candidates.extend(glob.glob(str(base / "**/final*.pt"), recursive=True))
        candidates.extend(glob.glob(str(base / "**/*.pt"), recursive=True))

    # prefer explicit best* names
    for c in candidates:
        if "best" in os.path.basename(c):
            return c
    if candidates:
        return candidates[0]
    return None

def _unpack_reset(reset_return):
    if isinstance(reset_return, tuple):
        return reset_return[0]
    return reset_return

def _unpack_step(step_return):
    # gym: (obs, reward, done, info)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if len(step_return) == 4:
        obs, reward, done, info = step_return
        truncated = info.get("TimeLimit.truncated", False) if isinstance(info, dict) else False
        done = done or truncated
        return obs, reward, done, info
    elif len(step_return) == 5:
        obs, reward, terminated, truncated, info = step_return
        done = terminated or truncated
        return obs, reward, done, info
    else:
        raise RuntimeError(f"Unexpected env.step() return length: {len(step_return)}")

def evaluate_best_model(
    model_path: str,
    EnvClass,
    eval_episodes: int = 20,
    seeds: int = 3,
    deterministic: bool = True,
    render: bool = False,
    base_seed: int = 12345
) -> Tuple[List[dict], dict]:
    # Import the PolicyNetwork from your training file (original signature)
    from training.reinforce_training import PolicyNetwork  # type: ignore

    # create a temp env to get observation/action sizes
    temp_env = EnvClass(render_mode=None)
    obs_space = temp_env.observation_space
    try:
        obs_size = obs_space.shape[0]
    except Exception:
        obs_size = int(np.prod(obs_space.shape)) if hasattr(obs_space, "shape") else 1
    n_actions = temp_env.action_space.n
    temp_env.close()

    device = torch.device("cpu")
    policy = PolicyNetwork(obs_size, n_actions).to(device)

    # load state dict (assumes saved via torch.save(policy.state_dict()))
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    results = []
    run_index = 0

    for seed_idx in range(seeds):
        seed_base = base_seed + seed_idx * 1000
        for ep in range(eval_episodes):
            run_seed = seed_base + ep
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            env = EnvClass(render_mode="human" if render else None)
            env = Monitor(env)

            reset_return = env.reset(seed=run_seed)
            obs = _unpack_reset(reset_return)
            ep_reward = 0.0
            steps = 0
            done = False
            info = {}

            while not done:
                # flatten dict observations if needed (simple concatenation)
                if isinstance(obs, dict):
                    vals = []
                    for v in obs.values():
                        arr = np.array(v)
                        vals.append(arr.ravel())
                    obs_arr = np.concatenate(vals).ravel()
                else:
                    obs_arr = np.array(obs).ravel()

                obs_tensor = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = policy(obs_tensor).squeeze(0)
                    # fallback: if outputs not normalized, softmax
                    if (out < 0).any() and not torch.isclose(out.sum(), torch.tensor(1.0)):
                        out = torch.softmax(out, dim=-1)

                    if deterministic:
                        action = int(torch.argmax(out).item())
                    else:
                        cat = torch.distributions.Categorical(out)
                        action = int(cat.sample().item())

                step_return = env.step(action)
                obs, reward, done, info = _unpack_step(step_return)

                ep_reward += float(reward)
                steps += 1

                if render:
                    try:
                        env.render()
                    except Exception:
                        pass

            # gather termination info if available
            terminated = bool(info.get("terminated", False)) if isinstance(info, dict) else False
            truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False

            results.append({
                "run_index": run_index,
                "seed_idx": seed_idx,
                "episode": ep,
                "seed": run_seed,
                "reward": float(ep_reward),
                "length": int(steps),
                "terminated": bool(terminated),
                "truncated": bool(truncated)
            })
            run_index += 1
            env.close()

    rewards = np.array([r["reward"] for r in results])
    lengths = np.array([r["length"] for r in results])
    terminated = np.array([r["terminated"] for r in results])
    truncated = np.array([r["truncated"] for r in results])

    summary = {
        "episodes_total": int(len(results)),
        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "reward_std": float(rewards.std(ddof=0)) if rewards.size else 0.0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "length_std": float(lengths.std(ddof=0)) if lengths.size else 0.0,
        "terminated_count": int(terminated.sum()),
        "truncated_count": int(truncated.sum()),
        "terminated_frac": float(terminated.mean()) if terminated.size else 0.0,
        "truncated_frac": float(truncated.mean()) if truncated.size else 0.0,
        "deterministic": bool(deterministic)
    }

    return results, summary

def save_mean_to_file(mean_value: float, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{mean_value}\n")
    print(f"Saved mean reward {mean_value:.6f} to {out_path}")

def save_results_csv(results: List[dict], out_path: str):
    if not results:
        print("No results to save.")
        return
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Write CSV with the specific columns the user expects
    fieldnames = ["seed_idx", "episode", "seed", "reward", "length", "terminated", "truncated"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"Saved per-episode results to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved .pt model (best_model.pt preferred).")
    parser.add_argument("--env-class", type=str, default="environment.custom_env.SocraticTutorEnv")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic argmax policy (default True).")
    parser.add_argument("--render", action="store_true", help="Render env during evaluation.")
    parser.add_argument("--out-file", type=str, default="best_model_mean.txt", help="File to write the mean reward to.")
    parser.add_argument("--out-csv", type=str, default="results/reinforce_eval_results.csv", help="CSV file to write per-episode results to.")
    args = parser.parse_args()

    model_path = find_reinforce_model(args.model_path)
    if model_path is None:
        raise FileNotFoundError("Could not find a REINFORCE model. Check --model-path or ../models/reinforce_engagement/")

    EnvClass = load_env_class(args.env_class)

    results, summary = evaluate_best_model(
        model_path,
        EnvClass,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        deterministic=args.deterministic,
        render=args.render
    )

    print("=== REINFORCE Evaluation Summary ===")
    print(f"Model path: {model_path}")
    print(f"Total episodes: {summary['episodes_total']}")
    print(f"Reward mean: {summary['reward_mean']:.6f}  std: {summary['reward_std']:.6f}")

    # Save only the mean value as requested
    save_mean_to_file(summary["reward_mean"], args.out_file)
    # Save per-episode CSV (the content you posted)
    save_results_csv(results, args.out_csv)

if __name__ == "__main__":
    main()