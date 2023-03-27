# University of Limerick
# Machine Learning - Methods and Applications
# Project: Deep Reinforcement Learning
# Author: Johannes Ernst, ID: 22387331

# Dependencies:
# Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
#   --> "MSVCv142 - VS 2019 C++ x64/x86 build tools" or greater
# pip install gym
# pip install gym[box2d]
# pip install tensorflow=2.12.0
# pip install keras=2.12.0
# pip install keras-rl2

# Import modules and libraries
import gym
import random

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam


# Build deep learning network model
def build_model(states_num, actions_num):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states_num)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions_num, activation='linear'))
    return model


# Build random environment
def run_random(env, episodes):
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


# # Build dqn agent with keras
# def build_agent(model, actions_num):

#     return dqn


if __name__ == '__main__':

    # Setting up the environment and extracting important variables
    env = gym.make("LunarLander-v2", 
               render_mode = "human",
               continuous = False,
               gravity = -7,
               enable_wind = False,
               wind_power = 15,
               turbulence_power = 1.5,
    )
    states_num = env.observation_space.shape[0]
    actions_num = env.action_space.n

    # Run environment with random actions 
    episodes = 3
    run_random(env, episodes)

    # Create deep learning network model
    model = build_model(states_num, actions_num)
    print(model.summary())

    # Build agent


    print("Succesful termination")