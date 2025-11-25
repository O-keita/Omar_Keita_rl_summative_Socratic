import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.rendering import Renderer

class SocraticTutorEnv(gym.Env):
    """
    Mission-based RL environment: AI Tutor guides a student
    from passive learning to mastery using Socratic methods.

    Notes on changes:
    - Adds per-episode hidden student_skill (0..1) to vary responsiveness.
    - Adds stochastic noise to action effects each step.
    - Adds diminishing returns for repeating the same action.
    - Rewards are based on measured improvement (delta engagement / delta confusion),
      and a bonus for state progression, instead of only per-action constants.
    - Does not change any public names or the structure of the observation/state dict,
      so existing rendering that relies on _get_state_dict(), state fields, etc. will still work.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    STATE_TYPES = ["Passive", "Confused", "Engaged", "Reflecting", "Mastery"]

    def __init__(self, render_mode="human"):
        super(SocraticTutorEnv, self).__init__()

        # Observation: [student_state_id, engagement, confusion, effort]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([4, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: 0-4
        self.action_space = spaces.Discrete(5)
        self.state = None
        self.steps = 0
        self.max_steps = 50
        self.render_mode = render_mode

        # Initialize Renderer
        self.renderer = Renderer() if render_mode == "human" else None

        # Randomness: use numpy Generator for reproducible noise when seed is provided in reset
        self.rng = np.random.default_rng()

        # Hidden per-episode parameters (do not expose in observation)
        self.student_skill = 0.5  # higher -> more receptive to effective actions
        self.repeat_count = 0
        self.last_action = None

    def _get_state_dict(self):
        """Convert state array to dict for rendering"""
        return {
            "student_state_id": int(self.state[0]),
            "engagement": float(self.state[1]),
            "confusion": float(self.state[2]),
            "effort": float(self.state[3])
        }

    def reset(self, seed=None, options=None):
        # Seed RNG for reproducibility when a seed is passed
        if seed is not None:
            # numpy.default_rng accepts an integer seed
            try:
                self.rng = np.random.default_rng(seed)
            except Exception:
                # fallback to unseeded generator if seed invalid
                self.rng = np.random.default_rng()

        # Initialize per-episode hidden student characteristics
        # student_skill controls how strongly the student responds to helpful actions (0..1)
        self.student_skill = float(self.rng.uniform(0.0, 1.0))

        # Reset repeat/action history
        self.repeat_count = 0
        self.last_action = None

        super().reset(seed=seed)

        # Initial student state: Passive, low engagement, high confusion, low effort
        # Add a small randomized jitter so episodes aren't identical
        engagement = 0.2 + self.rng.normal(0.0, 0.03)
        confusion = 0.8 + self.rng.normal(0.0, 0.03)
        effort = 0.2 + self.rng.normal(0.0, 0.03)

        self.state = np.array([
            0,
            float(np.clip(engagement, 0.0, 1.0)),
            float(np.clip(confusion, 0.0, 1.0)),
            float(np.clip(effort, 0.0, 1.0))
        ], dtype=np.float32)

        self.steps = 0

        # Render initial state
        if self.renderer:
            self.renderer.render(self._get_state_dict(), action=-1, step=0, reward=0)

        return self.state, {}

    def step(self, action):
        student_state_id, engagement, confusion, effort = self.state
        reward = 0.0

        # Record previous values to compute delta-based reward
        prev_engagement = float(engagement)
        prev_confusion = float(confusion)
        prev_state_id = int(student_state_id)

        # --- Handle repeat-action diminishing returns ---
        if action == self.last_action:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_action = action

        repeat_penalty_factor = 1.0 / (1.0 + 0.35 * self.repeat_count)  # reduces effect when repeating

        # Per-action nominal effects (base deltas)
        # Keep signs roughly same as before but scale them and make them influenceable by student_skill
        base_effects = {
            0: {"eng": 0.08,  "conf": 0.04,  "eff": 0.05},   # Ask Socratic Question
            1: {"eng": 0.04,  "conf": -0.05, "eff": 0.05},   # Give Hint
            2: {"eng": -0.03, "conf": -0.09, "eff": -0.04},  # Provide Code Example
            3: {"eng": 0.09,  "conf": -0.04, "eff": 0.09},   # Encourage Reflection
            4: {"eng": 0.05,  "conf": -0.05, "eff": 0.06},   # Ask Student to Explain
        }

        eff = base_effects.get(int(action), {"eng": 0.0, "conf": 0.0, "eff": 0.0})

        # Student responsiveness multiplier: higher skill -> stronger positive response, but also faster forgetting
        responsiveness = 0.4 + 0.8 * self.student_skill  # in ~[0.4, 1.2]

        # Add some stochastic noise to effects so environment is not fully deterministic
        noise_eng = float(self.rng.normal(0.0, 0.03))
        noise_conf = float(self.rng.normal(0.0, 0.03))
        noise_eff = float(self.rng.normal(0.0, 0.02))

        # Apply modified deltas
        engagement += (eff["eng"] * responsiveness * repeat_penalty_factor) + noise_eng
        confusion += (eff["conf"] * responsiveness * repeat_penalty_factor) + noise_conf
        effort += (eff["eff"] * responsiveness * repeat_penalty_factor) + noise_eff

        # Clamp values between 0-1
        engagement = np.clip(engagement, 0.0, 1.0)
        confusion = np.clip(confusion, 0.0, 1.0)
        effort = np.clip(effort, 0.0, 1.0)

        # --- Update student state based on engagement/confusion (same thresholds) ---
        if engagement > 0.8 and confusion < 0.3:
            student_state_id = 4  # Mastery
        elif engagement > 0.6:
            student_state_id = 3  # Reflecting
        elif engagement > 0.3:
            student_state_id = 2  # Engaged
        elif confusion > 0.6:
            student_state_id = 1  # Confused
        else:
            student_state_id = 0  # Passive

        self.state = np.array([student_state_id, engagement, confusion, effort], dtype=np.float32)
        self.steps += 1

        # --- Reward shaping based on actual improvement/deterioration ---
        delta_eng = float(self.state[1]) - prev_engagement
        delta_conf = prev_confusion - float(self.state[2])  # positive if confusion decreased
        # Encourage engagement increases and confusion decreases; small penalty for high confusion
        reward = 50.0 * delta_eng + 40.0 * delta_conf

        # Bonus for discrete state progression (e.g., moving from Confused->Engaged or to Mastery)
        if int(student_state_id) > prev_state_id:
            reward += 20.0 * (int(student_state_id) - prev_state_id)

        # Penalty if confusion becomes very high or engagement very low
        if float(self.state[2]) > 0.9:
            reward -= 5.0
        if float(self.state[1]) < 0.05:
            reward -= 8.0

        # Small per-step time penalty to encourage efficient teaching
        reward -= 0.1

        # --- Termination conditions ---
        terminated = int(student_state_id) == 4 or float(engagement) < 0.01
        truncated = self.steps >= self.max_steps

        # Render the current step
        if self.renderer:
            # provide a nicely rounded reward for rendering
            self.renderer.render(self._get_state_dict(), int(action), self.steps, float(round(reward, 3)))

        return self.state, float(reward), bool(terminated), bool(truncated), {}

    def render(self, mode="human"):
        if self.renderer:
            self.renderer.render(self._get_state_dict(), action=-1, step=self.steps, reward=0)

    def close(self):
        if self.renderer:
            self.renderer.close()