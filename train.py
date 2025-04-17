import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Assault-v5")

env.close()
