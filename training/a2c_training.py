from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SocraticTutorEnv

def make_env():
    env = SocraticTutorEnv(render_mode=None)
    env = Monitor(env)
    return env

if __name__ == "__main__":

    env = make_vec_env(make_env, n_envs=1)
    save_path = "../models/a2c_engagement"

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=save_path)

    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="../logs/",
        eval_freq=30_000,
        deterministic=True
    )

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.00001,
        gamma=0.99,
        n_steps=50,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5, 
        tensorboard_log="../tensorboard/",
        verbose=1
    )

    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save(f"{save_path}/final_model")
    print("Training complete!")
