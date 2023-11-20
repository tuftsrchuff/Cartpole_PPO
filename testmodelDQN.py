import sys
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN

#Run all saved models for 100 episodes and return average rewards
env = gym.make("CartPole-v1", render_mode="rgb_array")

models = 200
f = open(f"./logs/DQN_rewards.txt", "w")
for model_num in range(1,models+1):
    model_string = f"./logs/DQN_model_{5000*model_num}_steps.zip"
    model = DQN.load(model_string, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)
    print(f"Mean:{mean_reward} Std:{std_reward}")

    f.write(str(mean_reward))
    f.write("\n")
    del model

f.close()

env.close()