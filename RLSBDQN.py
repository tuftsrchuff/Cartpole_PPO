import sys
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN


env = gym.make("CartPole-v1", render_mode="rgb_array")
model = DQN("MlpPolicy", env, verbose=1, seed=100)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                         name_prefix='DQN_model')


model.learn(total_timesteps=1000000, callback=checkpoint_callback, progress_bar=True)