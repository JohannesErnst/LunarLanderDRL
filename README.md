# Lunar Lander Deep Reinforcement Learning
Train a deep reinforcement learning model to solve the Open AI Lunar Lander problem.

![crashgit](https://user-images.githubusercontent.com/51992212/229205576-242e8912-2b61-4fef-939c-0a6e5375a2a9.gif)
![landgit](https://user-images.githubusercontent.com/51992212/229205587-7c00a0a8-0842-4d78-90f5-b59bed0c97f0.gif)

## Environment
OpenAI Gym library [Lunar Lander](https://www.gymlibrary.dev/environments/box2d/lunar_lander/). <br>
Code is mainly based on the following repos:
- [Reinforcement-Learning-Lunar_Lander](https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander)
- [TensorflowKeras-ReinforcementLearning](https://github.com/nicknochnack/TensorflowKeras-ReinforcementLearning)

## Dependencies and Installation
Step 1:
- [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/): &rarr; "MSVCv142 - VS 2019 C++ x64/x86 build tools" or greater
- `pip install gym`
- `pip install gym[box2d]`
- `pip install tensorflow==2.12.0`
- `pip install keras==2.12.0`
- `pip install keras-rl2`

Step 2 (for DQNAgent Model):  <br>
Since the keras rl library (`keras-rl2`) is outdated and not updated anymore, some changed had to be made to the existing library files. In `...site-packages\rl` replace the `callbacks.py` and `core.py` file with the files in this repo. Same for `dqn.py` in `...site-packages\rl\agents`.

## How to use
Code provides two classes/models for training the Open AI Lunar Lander: DQNAgent (from keras rl library) and custom DQN model (based on keras library). Both models use a greedy policy (more information at [Reinforcement-Learning-Lunar_Lander](https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander)). In the main file, the model as well as other options like rendering and testing can be selected.
