# SocraticLearn: Reinforcement Learning Socratic Tutor

**Student:** Omar Keita  
**Repository:** [https://github.com/O-keita/SocraticLearn.git](https://github.com/O-keita/SocraticLearn.git)  
**Video Demonstration:** [https://youtu.be/1quyeS90zDI](https://youtu.be/1quyeS90zDI)

## Project Overview

SocraticLearn is an intelligent tutoring system powered by reinforcement learning that adapts teaching strategies in real-time to maximize student learning outcomes. The system uses a custom Gymnasium environment to simulate dynamic student states (engagement, confusion, effort) and trains RL agents to select optimal pedagogical interventions through trial and error.

This project compares four state-of-the-art RL algorithms—DQN, REINFORCE, A2C, and PPO—to determine which approach best models adaptive, personalized education.

## Environment Description

### Agent
An AI Socratic tutor that selects instructional actions to guide students through learning sessions, optimizing for engagement, reduced confusion, and increased effort.

### Action Space (Discrete - 5 Actions)

| Action ID | Description | Effect on Student |
|-----------|-------------|-------------------|
| 0 | Ask Socratic question | ↑ Engagement, ↑ Confusion, ↑ Effort |
| 1 | Give hint | ↑ Engagement, ↓ Confusion, ↑ Effort |
| 2 | Provide code example | ↓ Engagement, ↓ Confusion, ↓ Effort |
| 3 | Encourage reflection | ↑↑ Engagement, ↓ Confusion, ↑↑ Effort |
| 4 | Ask student to explain | ↑ Engagement, ↓ Confusion, ↑ Effort |

**Note:** Actual effects are modulated by hidden student skill, stochastic noise, and diminishing returns for action repetition.

### Observation Space
- **Engagement level** (0–1)
- **Confusion level** (0–1)
- **Effort level** (0–1)
- **Current performance score**
- **Steps taken in session**

### Reward Structure
Dynamic reward system based on **actual learning improvement**:

**Formula:**
```
Reward = 50×ΔEngagement + 40×ΔConfusion_Reduction + 20×State_Progression 
         - Repetition_Penalty - Time_Penalty + Stochastic_Noise
```

**Delta-Based Rewards:**
- `+50 × Δ(engagement)` - Rewards increased student engagement
- `+40 × Δ(confusion reduction)` - Rewards decreased confusion
- `+20 × state progression` - Bonus for advancing learning states (e.g., Confused → Engaged → Mastery)

**Penalties:**
- `-5.0` if confusion > 0.9 (student overwhelmed)
- `-8.0` if engagement < 0.05 (student disengaged)
- `-0.1` per step (time penalty for teaching efficiency)
- Diminishing returns for repeating the same action

**Key Features:**
- Rewards reflect measured pedagogical impact, not fixed action values
- Stochastic noise adds environmental uncertainty
- Hidden per-episode student skill parameter varies responsiveness
- Non-deterministic environment encourages robust policy learning

## Project Structure

```
SocraticLearn/
├── assets/                    # Student and teacher sprites
├── demos/                     # Visualization demos
│   ├── dqn_visualization.py
│   └── random_demo.py
├── environment/               # Custom Gymnasium environment
│   ├── custom_env.py
│   └── rendering.py
├── evaluate/                  # Model evaluation scripts
│   ├── dqn_evaluate.py
│   ├── a2c_evaluate.py
│   ├── ppo_evaluate.py
│   └── reinforce_evaluate.py
├── training/                  # Training scripts
│   ├── dqn_training.py
│   ├── a2c_training.py
│   ├── ppo_training.py
│   └── reinforce_training.py
├── models/                    # Saved trained models
│   ├── dqn_engagement/
│   ├── a2c_engagement/
│   ├── ppo_engagement/
│   └── reinforce_engagement/
├── tensorboard/               # Training logs
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md
```

## Environment Visualization

### Demo Video
Watch the environment in action with random and trained agents:

**Static Random Agent Demo:**  
[Environment Visualization - Random Actions](https://github.com/O-keita/SocraticLearn.git)

### Visual Elements
The environment features:
- **Teacher sprite** - Represents the AI Socratic tutor agent
- **Student sprites** (5 states) - Visual representation of learning progression:
  - **Passive** - Low engagement, high confusion
  - **Confused** - Struggling with concepts
  - **Engaged** - Actively participating
  - **Reflecting** - Deep thinking and metacognition
  - **Mastery** - Goal state, comprehensive understanding
- **Real-time metrics display** - Shows engagement, confusion, and effort levels
- **Action feedback** - Visual indicators of tutor interventions
- **State transitions** - Smooth animations between learning states

The visualization is built using **Pygame**, providing an intuitive view of how different teaching strategies affect student learning dynamics in real-time.

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/O-keita/SocraticLearn.git
cd SocraticLearn
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Best Model
Execute the main script to run the best-performing trained agent:
```bash
python main.py
```

### Training Models

Train individual algorithms:
```bash
# DQN
python training/dqn_training.py

# REINFORCE
python training/reinforce_training.py

# A2C
python training/a2c_training.py

# PPO
python training/ppo_training.py
```

### Evaluating Models
```bash
python evaluate/dqn_evaluate.py
python evaluate/reinforce_evaluate.py
python evaluate/a2c_evaluate.py
python evaluate/ppo_evaluate.py
```

### Visualization Demos
```bash
# Random agent demo
python demos/random_demo.py

# DQN trained agent visualization
python demos/dqn_visualization.py
```

### Monitoring Training (TensorBoard)
```bash
tensorboard --logdir=./tensorboard
```
Open http://localhost:6006 in your browser.

## Algorithms Implemented

### 1. Deep Q-Network (DQN)
**Value-based method** using experience replay and target networks.

**Architecture:**
- MLP policy with 2 hidden layers (Stable-Baselines3 default)
- Experience replay buffer to break correlation between steps
- Target network updated every 10k steps for stability
- Epsilon-greedy exploration (ε: 0.2 → 0.02)
- Trains every 4 steps with batch size 32

**Best Hyperparameters:**
- Learning Rate: 0.001
- Gamma: 0.99
- Replay Buffer: 200,000
- Batch Size: 32
- Epsilon: 0.1 → 0.02
- **Mean Reward: 293.62**

### 2. REINFORCE
**Policy gradient method** using Monte Carlo returns.

**Architecture:**
- Direct policy optimization without value function
- High variance gradient estimates
- Episode-based updates

**Best Hyperparameters:**
- Learning Rate: 0.003
- Gamma: 0.995
- Episodes: 5000
- **Mean Reward: 319**

**Note:** Highly unstable - most runs collapsed to negative rewards despite peak performance.

### 3. Advantage Actor-Critic (A2C)
**Actor-critic method** with parallel environments.

**Architecture:**
- MLP policy with 2 hidden layers
- Synchronous policy updates
- Advantage estimation for variance reduction
- Entropy bonus for exploration

**Best Hyperparameters:**
- Learning Rate: 0.0007
- Gamma: 0.99
- Entropy Coefficient: 0.01
- n_steps: 5
- GAE Lambda: 0.95
- **Mean Reward: 133.59**

### 4. Proximal Policy Optimization (PPO)
**Advanced policy gradient** with clipped surrogate objective.

**Architecture:**
- MLP policy with 2 hidden layers
- Stochastic policy with action probabilities
- GAE (λ=0.99) for advantage estimation
- Multiple parallel environments (n_envs=4)
- Mini-batch updates with multiple epochs

**Best Hyperparameters:**
- Learning Rate: 0.0003
- n_steps: 50
- Clip Range: 0.2
- Entropy Coefficient: 0.01
- Gamma: 0.99
- **Mean Reward: 324.33**

## Key Results

### Performance Comparison

| Algorithm | Mean Reward | Training Stability | Convergence Speed | Generalization |
|-----------|-------------|-------------------|-------------------|----------------|
| **PPO** | **324.33** | ⭐⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐⭐ Fast (~250k steps) | ⭐⭐⭐⭐⭐ Excellent |
| **REINFORCE** | **319** | ⭐⭐ Very Low | ⭐⭐ Slow | ⭐ Poor |
| **DQN** | **293.62** | ⭐⭐⭐⭐ High | ⭐⭐⭐ Medium (~280k steps) | ⭐⭐⭐ Moderate |
| **A2C** | **133.59** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Fast | ⭐ Poor |

### Key Findings

**PPO - Best Overall Performance:**
- Achieved highest stable mean reward (324.33)
- Most consistent learning with lowest variance
- Converged fastest (~250k timesteps to plateau)
- **Best generalization** to unseen student states
- Maintained high rewards across diverse initial conditions
- Smooth, reliable policy updates thanks to clipped objective

**DQN - Strong but Less Robust:**
- Competitive performance (293.62 mean reward)
- Gradual improvement with noticeable fluctuations
- Required more training time (~280k-300k timesteps)
- **Moderate generalization** - worked well on similar states but struggled with high confusion scenarios
- Some evidence of overfitting to training distribution

**REINFORCE - High Variance, Unreliable:**
- Achieved second-highest peak reward (319) in best configuration
- **Highly unstable** - majority of runs collapsed to negative rewards
- Severe variance in gradient estimates
- **Poor generalization** - failed on difficult initial conditions
- Lacks stabilizing mechanisms (baseline, value function)
- Not suitable for production use despite peak performance

**A2C - Failed to Learn Effectively:**
- Lowest performance (133.59 mean reward)
- Plateaued quickly around 118-130 reward
- Failed to leverage advantage estimation effectively
- **No meaningful generalization**
- Low sample efficiency in this environment
- Did not improve significantly over training

### Training Dynamics

**Cumulative Reward Progression:**
- **PPO**: Consistent increase, stabilized at 350-380 cumulative reward by 250k steps
- **DQN**: Gradual climb with fluctuations, plateau at 300-320 by 300k steps  
- **REINFORCE**: Erratic trajectory, frequent collapses despite occasional peaks
- **A2C**: Quick plateau with minimal improvement beyond early training

## Hyperparameter Tuning

The project includes extensive hyperparameter exploration with **10+ configurations per algorithm**:

### DQN (10 configurations)
Key parameters explored:
- Learning rates: 0.00001 - 0.001
- Replay buffer sizes: 100k - 300k
- Batch sizes: 32, 64
- Epsilon decay strategies: Various start/end combinations

### REINFORCE (8 configurations)
Key parameters explored:
- Learning rates: 0.00001 - 0.05
- Gamma: 0.99 - 0.995
- Total episodes: 1000 - 7000

### A2C (11 configurations)
Key parameters explored:
- Learning rates: 0.00001 - 0.001
- Entropy coefficients: 0.001 - 0.03
- n_steps: 5 - 50
- GAE lambda: 0.90 - 0.99
- Gamma: 0.70 - 0.99

### PPO (10 configurations)
Key parameters explored:
- Learning rates: 0.00001 - 0.0003
- n_steps: 50 - 512
- Clip ranges: 0.2 - 0.3
- Entropy coefficients: 0.01 - 0.1
- Gamma: 0.95 - 0.995

**See full hyperparameter tables in the project report for detailed results.**

## Dependencies

Key libraries:
- `gymnasium` - Environment framework
- `stable-baselines3` - RL algorithms
- `torch` - Neural networks
- `pygame` - Visualization
- `tensorboard` - Training monitoring
- `numpy`, `matplotlib` - Data processing and visualization

See `requirements.txt` for complete list.

## Video Demonstration

**3-Minute Project Demonstration:** [https://youtu.be/1quyeS90zDI](https://youtu.be/1quyeS90zDI)

The video includes:
- Problem statement and project overview
- Agent behavior and learning dynamics
- Reward structure explanation
- Best-performing model (PPO) in action
- GUI visualization and terminal outputs
- Performance analysis and results discussion

## Conclusion

### Summary of Findings

**PPO emerged as the best algorithm** for the Socratic Tutor environment, demonstrating:
- Highest stable performance (324.33 mean reward)
- Superior training stability with minimal variance
- Fastest convergence (~250k timesteps)
- Excellent generalization to unseen student states
- Robust performance across diverse initial conditions

**DQN performed competitively** but required more training time and showed moderate overfitting to training conditions, particularly struggling with high-confusion scenarios.

**REINFORCE achieved impressive peak performance** (319 mean reward) but proved unreliable due to extreme variance and frequent training collapses, making it unsuitable for practical deployment.

**A2C failed to learn an effective policy**, plateauing early with poor sample efficiency and no meaningful generalization.

### Strengths and Weaknesses

| Algorithm | Strengths | Weaknesses |
|-----------|-----------|------------|
| **PPO** | Stable, fast convergence, excellent generalization | Slightly more complex implementation |
| **DQN** | Reliable, interpretable Q-values | Needs more training, some overfitting |
| **REINFORCE** | Simple, can achieve high peaks | Unstable, high variance, unreliable |
| **A2C** | Fast initial learning | Poor performance, no generalization |

### Why PPO Excels in This Environment

The Socratic Tutor environment presents several challenges that favor PPO:
1. **Non-deterministic dynamics** - Student responses vary stochastically; PPO's clipped objective prevents destructive policy updates
2. **Sparse state progression** - GAE helps credit assignment across long horizons
3. **Action repetition penalties** - PPO's exploration bonus (entropy) encourages diverse strategies
4. **Hidden student skill parameter** - PPO generalizes better to unseen student types

### Future Improvements

With additional time and resources:
- **Curriculum learning** - Start with easier student profiles, gradually increase difficulty
- **Reward shaping refinement** - Experiment with different delta coefficients
- **Hierarchical policies** - Separate strategy selection from action execution
- **Multi-student environments** - Train on diverse student archetypes simultaneously
- **Real student data integration** - Validate on actual learning trajectories
- **Longer time horizons** - Test policies over multi-session learning sequences

---

**For detailed experimental results, training curves, and complete analysis, refer to the full project report.**