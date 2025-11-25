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

# stable-baselines3
from stable_baselines3 import PPO

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

def find_model(model_path_hint: str):
    if model_path_hint:
        p = Path(model_path_hint)
        if p.exists():
            return str(p)
    base = Path("../models/ppo_engagement")
    candidates = []
    if base.exists():
        candidates.extend(glob.glob(str(base / "best_model*")))
        candidates.extend(glob.glob(str(base / "best_model*.zip")))
        candidates.extend(glob.glob(str(base / "final_model*")))
        candidates.extend(glob.glob(str(base / "*.zip")))
    for c in candidates:
        if "best_model" in os.path.basename(c):
            return c
    if candidates:
        return candidates[0]
    return None

def evaluate_model(model_path, EnvClass, eval_episodes=20, seeds=3, render=False):
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    results = []
    for seed_idx in range(seeds):
        seed_base = 100000 + seed_idx * 1000
        for ep in range(eval_episodes):
            env = EnvClass(render_mode="human" if render else None)
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)

            obs, info = env.reset(seed=seed_base + ep)
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                steps += 1
                done = terminated or truncated

            results.append({
                "seed_idx": seed_idx,
                "episode": ep,
                "reward": float(ep_reward),
                "length": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            env.close()

    rewards = np.array([r["reward"] for r in results])
    lengths = np.array([r["length"] for r in results])

    summary = {
        "episodes_total": len(results),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
    }

    return results, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--env-class", type=str, default="environment.custom_env.SocraticTutorEnv")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--render", type=lambda x: x.lower() in ("1","true","yes"), default=False)
    parser.add_argument("--out-csv", type=str, default="ppo_eval_results.csv")
    args = parser.parse_args()

    model_path = find_model(args.model_path)
    if not model_path:
        raise FileNotFoundError("No PPO model found")

    EnvClass = load_env_class(args.env_class)
    results, summary = evaluate_model(
        model_path,
        EnvClass,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        render=args.render
    )

    print("=== PPO Evaluation Summary ===")
    print(f"Mean reward: {summary['reward_mean']:.2f}  std: {summary['reward_std']:.2f}")
    print(f"Mean length: {summary['length_mean']:.2f}  std: {summary['length_std']:.2f}")

if __name__ == "__main__":
    main()
