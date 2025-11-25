import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SocraticTutorEnv

# -------------------------------
# Policy Network
# -------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# REINFORCE TRAINING
# -------------------------------
def train_reinforce(
    env,
    policy,
    optimizer,
    gamma=0.99,
    total_episodes=1000,
    save_path="../models/reinforce_engagement"
):

    os.makedirs(save_path, exist_ok=True)

    best_mean_reward = -999
    rewards_history = []

    for episode in range(total_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        ep_reward = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_t)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            ep_reward += reward
            done = terminated or truncated

        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(ep_reward)
        mean_reward = np.mean(rewards_history[-50:])

        print(f"Episode {episode+1}/{total_episodes} | Reward: {ep_reward:.2f} | Mean50: {mean_reward:.2f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(policy.state_dict(), f"{save_path}/best_model.pt")

    torch.save(policy.state_dict(), f"{save_path}/final_model.pt")
    print("Training complete!")

# -------------------------------
# MAIN (same flow as PPO)
# -------------------------------
if __name__ == "__main__":

    env = Monitor(SocraticTutorEnv(render_mode=None))

    save_path = "../models/reinforce_engagement"

    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-2)

    train_reinforce(
        env,
        policy,
        optimizer,
        gamma=0.99,
        total_episodes=7000,
        save_path=save_path
    )

    print("REINFORCE training complete!")
