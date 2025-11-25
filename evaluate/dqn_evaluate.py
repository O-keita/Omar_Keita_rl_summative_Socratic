#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
import glob
import csv
from pathlib import Path
import numpy as np
import gymnasium as gym

# stable-baselines3
from stable_baselines3 import DQN

def load_env_class(env_path: str):
    # env_path examples: "environment.custom_env.SocraticTutorEnv" or "environment.custom_env:SocraticTutorEnv"
    if ":" in env_path:
        module_path, cls_name = env_path.split(":")
    else:
        parts = env_path.split(".")
        module_path = ".".join(parts[:-1])
        cls_name = parts[-1]
    module = importlib.import_module(module_path)
    EnvClass = getattr(module, cls_name)
    return EnvClass

def find_model(model_path_hint: str):
    # Try provided path, then common defaults.
    if model_path_hint:
        p = Path(model_path_hint)
        if p.exists():
            return str(p)
    base = Path("../models/dqn_engagement")
    candidates = []
    if base.exists():
        # common SB3 filenames used by EvalCallback/DQN.save
        candidates.extend(glob.glob(str(base / "best_model*")))
        candidates.extend(glob.glob(str(base / "best_model*.zip")))
        candidates.extend(glob.glob(str(base / "final_model*")))
        candidates.extend(glob.glob(str(base / "*.zip")))
    # pick best_model first, then any zip
    for c in candidates:
        if "best_model" in os.path.basename(c):
            return c
    if candidates:
        return candidates[0]
    return None

def evaluate_model(model_path, EnvClass, eval_episodes=20, seeds=3, render=False, render_delay=0.05):
    # Load model (DQN)
    print(f"Loading model from: {model_path}")
    model = DQN.load(model_path)

    results = []
    total_runs = seeds * eval_episodes
    for seed_idx in range(seeds):
        seed_base = 100000 + seed_idx * 1000
        for ep in range(eval_episodes):
            env = EnvClass(render_mode="human" if render else None)
            # wrap with Monitor for consistent logging (not strictly required)
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)

            seed = seed_base + ep
            obs, info = env.reset(seed=seed)
            ep_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += float(reward)
                steps += 1
                if render:
                    # allow the environment's renderer to show state; the env's own render handles fps
                    try:
                        env.render()
                    except Exception:
                        pass
                if terminated or truncated:
                    results.append({
                        "seed_idx": seed_idx,
                        "episode": ep,
                        "seed": seed,
                        "reward": float(ep_reward),
                        "length": int(steps),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated)
                    })
                    break
            env.close()

    # Summaries
    rewards = np.array([r["reward"] for r in results])
    lengths = np.array([r["length"] for r in results])
    terminated = np.array([r["terminated"] for r in results])
    truncated = np.array([r["truncated"] for r in results])

    summary = {
        "episodes_total": len(results),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "terminated_count": int(terminated.sum()),
        "truncated_count": int(truncated.sum()),
        "terminated_frac": float(terminated.mean()),
        "truncated_frac": float(truncated.mean())
    }

    return results, summary

def save_results_csv(results, out_path="dqn_eval_results.csv"):
    if not results:
        return
    keys = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-episode results to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model (.zip).")
    parser.add_argument("--env-class", type=str, default="environment.custom_env.SocraticTutorEnv",
                        help="Environment class import path.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--render", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
    parser.add_argument("--out-csv", type=str, default="eval_results.csv")
    args = parser.parse_args()

    model_path = find_model(args.model_path)
    if model_path is None:
        raise FileNotFoundError("Could not find a model. Check --model-path or ../models/dqn_engagement/")

    EnvClass = load_env_class(args.env_class)
    results, summary = evaluate_model(model_path, EnvClass, eval_episodes=args.eval_episodes, seeds=args.seeds, render=args.render)

    print("=== Evaluation Summary ===")
    print(f"Total episodes: {summary['episodes_total']}")
    print(f"Reward mean: {summary['reward_mean']:.2f}  std: {summary['reward_std']:.2f}")
    print(f"Length mean: {summary['length_mean']:.2f}  std: {summary['length_std']:.2f}")
    print(f"Terminated (Mastery) count: {summary['terminated_count']}  fraction: {summary['terminated_frac']:.2%}")
    print(f"Truncated count: {summary['truncated_count']}  fraction: {summary['truncated_frac']:.2%}")

    save_results_csv(results, args.out_csv)

if __name__ == "__main__":
    main()