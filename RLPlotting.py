import matplotlib.pyplot as plt

timesteps = []
A2Crewards = []
PPOrewards = []
DQNrewards = []
PPOrewards_mod = []
for i in range(0, 21):
    timesteps.append(i * 5000)

#A2C
# f = open("./logs/A2C_rewards.txt", "r")
# for x in f:
#   A2Crewards.append(float(x))

# f.close()

# plt.plot(timesteps, A2Crewards, marker = 'o', label ='A2C')

#PPO
f = open("./logs/PPO_rewards.txt", "r")
for x in f:
  PPOrewards.append(float(x))

f.close()
plt.plot(timesteps, PPOrewards, label ='PPO')


#DQN
# f = open("./logs/DQN_rewards.txt", "r")
# for x in f:
#   DQNrewards.append(float(x))

# f.close()

# plt.plot(timesteps, DQNrewards, marker = 'o', label ='DQN')

#PPO Mod
f = open("./logs/PPO_rewards_mod.txt", "r")
for x in f:
  PPOrewards_mod.append(float(x))

f.close()

plt.plot(timesteps, PPOrewards_mod, label ='PPO_mod')




plt.xlabel("Timesteps")
plt.ylabel("Mean reward per 100 episodes")
plt.title("Cartpole")
plt.legend()

plt.show()
