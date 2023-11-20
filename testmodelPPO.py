import sys
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

#Run all saved models for 100 episodes and return average rewards
env = gym.make("CartPole-v1", render_mode="rgb_array")

models = 20
save_freq = 5000
f = open(f"./logs/PPO_rewards.txt", "w")
for model_num in range(1,models+1):
    mean_reward_all_mod = 0
    for i in range(1, 6):
        model_string = f"./logs/PPO_model_{i}_{save_freq*model_num}_steps.zip"
        model = PPO.load(model_string, env=env)
        vec_env = model.get_env()
        obs = vec_env.reset()
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)
        print(f"Mean:{mean_reward} Std:{std_reward}")
        mean_reward_all_mod += mean_reward
        del model
    
    mean_reward_all_mod = mean_reward_all_mod / 5

    f.write(str(mean_reward_all_mod))
    f.write("\n")
    

f.close()

vec_env.close()
env.close()