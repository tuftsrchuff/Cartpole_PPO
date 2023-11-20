import sys
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")

seeds = [1, 2, 3, 4, 5]

for i in seeds:
    model = PPO("MlpPolicy", env, verbose=1, seed=i)
    # Save a checkpoint every 5000 steps = 20 models per seed
    prefix = f"PPO_model_{i}"
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                         name_prefix=prefix)

    model.learn(total_timesteps=100000, callback=checkpoint_callback, progress_bar=True)

