from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import pandas as pd
from indoor_explorers.envs.settings import DEFAULT_CONFIG as conf
from indoor_explorers.envs.explorerConf import ExplorerConf  

#%% Set up the environment
env = ExplorerConf(conf = conf)  

#%% Create a deep learning model with keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

states = env.observation_space.shape
actions = env.action_space.n

model = build_model(states, actions)
print(model.summary())

#%% Build Agent wit Keras-RL
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent (model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dqn = DQNAgent (model = model, memory = memory, policy=policy,
                    nb_actions=actions, nb_steps_warmup=10, target_model_update= 1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics = ['mae'])
dqn.fit (env, nb_steps = 4000, visualize=False, verbose = 1)