import os

# Define project structure
project_root = "socratic_learn_rl"

folders = [
    "environment",
    "training",
    "models"
]

files = {
    "environment/custom_env.py": "# Custom classroom RL environment\n",
    "environment/rendering.py": "# Simple rendering logic (text or 2D grid)\n",
    "training/dqn_training.py": "# DQN training script\n",
    "training/pg_training.py": "# Policy Gradient training script\n",
    "requirements.txt": "gymnasium\nstable-baselines3\ntorch\nnumpy\npygame\n",
    "README.md": "# SocraticLearn RL Environment\n\nA reinforcement learning simulation of Socratic classroom interactions.\n",
    "main.py": "# Entry point for running and testing the environment\n\nfrom environment.custom_env import ClassroomEnv\n\nif __name__ == '__main__':\n    env = ClassroomEnv()\n    obs, _ = env.reset()\n    print('Initial Observation:', obs)\n"
}

# Create folders
os.makedirs(project_root, exist_ok=True)
for folder in folders:
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)

# Create files
for path, content in files.items():
    full_path = os.path.join(project_root, path)
    with open(full_path, "w") as f:
        f.write(content)

print("✅ Project structure created successfully at:", project_root)
