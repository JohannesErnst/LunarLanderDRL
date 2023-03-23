# University of Limerick
# Machine Learning - Methods and Applications
# Project: Deep Reinforcement Learning
# Author: Johannes Ernst, ID: 22387331

# Dependencies:
# Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
#   --> "MSVCv142 - VS 2019 C++ x64/x86 build tools" or greater
# pip install gym
# pip install gym[box2d]

# Import modules and libraries
import gym
import random

# Setting up the environment and extracting variables
env = gym.make("LunarLander-v2", 
               render_mode = "human",
               continuous = False,
               gravity = -7,
               enable_wind = False,
               wind_power = 15,
               turbulence_power = 1.5,
)
states = env.observation_space.shape[0]
actions = env.action_space.n

# Build random environment
episodes = 3
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = random.choice([0,1,2,3])
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
    print('Episode: {} Score: {:.2f}'.format(episode, score))


print("Succesful termination")