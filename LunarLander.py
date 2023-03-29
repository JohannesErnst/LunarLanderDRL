# University of Limerick
# Machine Learning - Methods and Applications
# Project: Deep Reinforcement Learning
# Author: Johannes Ernst, ID: 22387331

# Dependencies:
# Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
#   --> "MSVCv142 - VS 2019 C++ x64/x86 build tools" or greater
# pip install gym
# pip install gym[box2d]
# pip install tensorflow==2.12.0
# pip install keras==2.12.0
# pip install keras-rl2

# Code mainly based on:
# https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander
# https://github.com/nicknochnack/TensorflowKeras-ReinforcementLearning


# Import modules and libraries
import gym
import random
import os
import sys

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


# Create a custom model with DQN network
class CustomModel:
    def __init__(self, env, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):

        # Environment variables
        self.env = env
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]
        self.counter = 0

        # Training variables
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.rewards_list = []
        
        # Set up architecture
        self.replay_memory_buffer = deque(maxlen=500000)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.num_observations, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_actions)

        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])
    
    def learn_weights(self):

        # Size check for replay_memory_buffer
        if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        # Select random samples from the replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        states = np.squeeze(np.squeeze(np.array([i[0] for i in random_sample])))
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.squeeze(np.array([i[3] for i in random_sample]))
        done_list = np.array([i[4] for i in random_sample])

        # Apply ϵ-greedy policy
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        # Update the weights
        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def train(self, num_episodes=2000):
        for episode in range(num_episodes):

            # Reset variables
            state = env.reset()[0]
            reward_for_episode = 0
            num_steps = 1000
            state = np.reshape(state, [1, self.num_observations])

            # Loop over maximum number of steps
            for step in range(num_steps):
                received_action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observations])

                # Store the experience in replay memory
                self.replay_memory_buffer.append((state, received_action, reward, next_state, done))
                
                # Add up rewards
                reward_for_episode += reward

                # Update the state and counter
                state = next_state
                self.update_counter()

                # Learn weights according to experience replay and ϵ-greedy policy
                self.learn_weights()

                if done:
                    break

            # Append the rewards list for this episode
            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 200:
                print("DQN Training Complete...")
                break
            print("{:.0f}\t: Episode || Reward: {:.3f} \t|| Average Reward: {:.3f} \t epsilon: {:.3f}"
                  .format(episode, reward_for_episode, last_rewards_mean, self.epsilon))
    
    def test(self, env, num_test_episode):

        # Initialize variables
        rewards_list = []
        step_count = 1000
        print("Starting Testing of the trained model...")

        # Loop over episodes
        for test_episode in range(num_test_episode):

            # Reset variables
            current_state = env.reset()[0]
            current_state = np.reshape(current_state, [1, self.num_observations])
            reward_for_episode = 0

            # Loop over maximum number of steps
            for step in range(step_count):
                selected_action = np.argmax(self.model.predict(current_state)[0])
                new_state, reward, done, _, _ = env.step(selected_action)
                new_state = np.reshape(new_state, [1, self.num_observations])
                current_state = new_state
                reward_for_episode += reward
                if done:
                    break

            # Append the rewards list for this episode
            rewards_list.append(reward_for_episode)
            print("{:.0f}\t: Episode || Reward: {:.3f}".format(test_episode, reward_for_episode))

        return rewards_list

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)


# Create a model for DQNAgent (keras rl library)
class DQNAgentModel:
    def __init__(self, env, lr=1e-3, epsilon=0.4):

        # Environment variables
        self.env = env
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]

        # Training variables
        self.lr = lr
        self.epsilon = epsilon
        
        # Set up architecture and agent
        self.replay_memory_buffer = deque(maxlen=500000)
        self.model = self.build_model()
        self.dqn = self.build_agent()
    
    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.num_observations)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        print(model.summary())
        return model
    
    def build_agent(self):

        # Set up policy and memory and initialize the agent
        policy = EpsGreedyQPolicy(eps=self.epsilon)
        memory = SequentialMemory(limit=500000, window_length=1)
        dqn = DQNAgent(model=self.model, memory=memory, policy=policy,
                       nb_actions=self.num_actions, nb_steps_warmup=10, target_model_update=1e-2)
        
        # Compile dqn and keep mean absolute error as metric
        dqn.compile(Adam(learning_rate=self.lr), metrics=['mae'])

        return dqn
    
    def train(self, env, num_episodes, visualize=False, verbose=1):
        self.dqn.fit(env, nb_steps=num_episodes, visualize=visualize, verbose=verbose)
    
    def test(self, env, num_test_episode):
        scores = self.dqn.test(env, nb_episodes=num_test_episode)
        print(np.mean(scores.history['episode_reward']))
    
    def save(self, filename, overwrite=True):
        self.dqn.save_weights(filename, overwrite=overwrite)
    
    def load(self, filename):
        self.dqn.load_weights(filename)


# Build environment and run with random decisions
def run_random(env, episodes):
    for episode in range(1, episodes+1):
        env.reset()
        terminated = False
        score = 0

        while not terminated:
            env.render()
            action = random.choice([0,1,2,3])
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
        print('Episode: {} Score: {:.2f}'.format(episode, score))
    sys.exit()


if __name__ == '__main__':

    # Select model (either 'custom', 'DQNAgent' or 'random')
    model_type = 'custom'

    # Select whether to train or test
    train = 0

    # Select whether to render or not
    render = 1

    # Filenames for models
    filename_dqn_agent = 'dqn_weights_trained.h5'
    filename_custom = 'model_trained.h5'

    # Folders for saving the models
    model_folder_dqn_agent = 'models_dqn_agent/'
    model_folder_custom = 'models_custom/'

    # Check whether the specified path exists or not
    if not os.path.exists(model_folder_dqn_agent):
        os.makedirs(model_folder_dqn_agent)
    if not os.path.exists(model_folder_custom):
        os.makedirs(model_folder_custom)

    # Setting up the environment and extracting important variables
    if render:
        env = gym.make("LunarLander-v2", render_mode = "human",
                       continuous = False, gravity = -10, 
                       enable_wind = False, wind_power = 15, turbulence_power = 1.5)
    else:
        env = gym.make("LunarLander-v2",
                       continuous = False, gravity = -10, 
                       enable_wind = False, wind_power = 15, turbulence_power = 1.5)

    # Create model
    if model_type == 'DQNAgent':
        DQN_model = DQNAgentModel(env, epsilon=0.1)
    elif model_type == 'custom':
        custom_model = CustomModel(env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995)
    elif model_type == 'random':
        run_random(env, episodes=5)
    else:
        raise ValueError('Model type ' + model_type + ' not supported!')

    if train:

        # Train the model
        if model_type == 'DQNAgent':
            DQN_model.train(env, num_episodes=60000)
            DQN_model.save(model_folder_dqn_agent + filename_dqn_agent)
            DQN_model.load(model_folder_dqn_agent + filename_dqn_agent)
            DQN_model.test(env, num_test_episode=20)
        else:
            custom_model.train(num_episodes=1000)
            custom_model.save(model_folder_custom + filename_custom)
            custom_model.load(model_folder_custom + filename_custom)
            custom_model.test(env, num_test_episode=20)

    else:

        # Test the model
        if model_type == 'DQNAgent':
            DQN_model.load(model_folder_dqn_agent + filename_dqn_agent)
            DQN_model.test(env, num_test_episode=20)
        else:
            custom_model.load(model_folder_custom + filename_custom)
            custom_model.test(env, num_test_episode=20)

    print("Succesful termination")