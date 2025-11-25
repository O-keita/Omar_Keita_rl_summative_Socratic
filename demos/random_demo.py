import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from environment.custom_env import SocraticTutorEnv
from environment.rendering import Renderer
import numpy as np

# Create the environment
env = SocraticTutorEnv()

# Initialize renderer
renderer = Renderer()

# Store metrics
engagement_list = []
confusion_list = []
effort_list = []
reward_list = []

obs, _ = env.reset()
done = False
step = 0

while not done:
    # TAKE A RANDOM ACTION
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # extract state values for visualization
    state = env._get_state_dict()
    engagement_list.append(state["engagement"])
    confusion_list.append(state["confusion"])
    effort_list.append(state["effort"])
    reward_list.append(reward)

    # Render the state visually
    renderer.render(state, action, step=step, reward=reward)
    step += 1
    renderer.clock.tick(5)   # slow down so you can see the animation

print("\n=== RANDOM POLICY SUMMARY ===")
print(f"Average Engagement: {np.mean(engagement_list):.2f}")
print(f"Average Confusion: {np.mean(confusion_list):.2f}")
print(f"Average Effort: {np.mean(effort_list):.2f}")
print(f"Total Reward: {np.sum(reward_list):.2f}")
