import gym
from gym import spaces
import retro
import numpy as np
import cv2
import random
from collections import deque
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from multi_agent.utils.multi_printMaps import *
# Discretize continuous action space
class Discretizer(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

# Limit the episode length
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action, PRINT)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

# Skip frames
class SkipFrames(gym.Wrapper):
    def __init__(self, env, n = 4):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totalReward = 0.0
        for _ in range(self.n):
            obs, reward, done, info = self.env.step(action)
            totalReward += reward
            print(totalReward)
            if done:
                break
        return obs, totalReward, done, info

# Convert observation to greyscale
class Rgb2Gray(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, _oldc) = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low = 0, high = 255,
                                                shape = (oldh, oldw, 1),
                                                dtype = np.uint8)
    
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:,:,None]

# Downsample the observation
class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh//ratio, oldw//ratio, oldc)
        self.observation_space = gym.spaces.Box(low = 0, high = 255,
                                                shape = newshape,
                                                dtype = np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame

#change observation space to return 4 stacked frames for temporal information
from collections import deque
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        (oldh, oldw, _oldc) = env.observation_space[0].shape #antes estava só observatio_space
        newStackShape = (oldh, oldw, k)
        # self.observation_space = gym.spaces.Box(low = 0, high = 255,
        #                                         shape = newStackShape,
        #                                         dtype = np.uint8)

        #Needs fixing:
        #highest value that can be observed in each cell is the max. agent id
        _obs_high = np.full((env.get_grid_shape()[0], env.get_grid_shape()[1],k), np.array(env.n_agents, dtype=np.float32)) 
        # #lowest value that can be observed in each cell is 0.0
        _obs_low = np.full((env.get_grid_shape()[0], env.get_grid_shape()[1],k),np.array(0.0, dtype=np.float32))
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(_obs_low, _obs_high, (env.get_grid_shape()[0], env.get_grid_shape()[1],k)) for _ in range(env.n_agents)])

        #self.state_shape = tf.reshape(self.env.observation_space, [self.state_shape[0],self.state_shape[1],k])
        self.k = k
        self.frames = deque([], maxlen = k)

    def reset(self):
        obs_n = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs_n[0].copy()) #antes estava só obs
        # for frame in self.frames:
        #     print(np.matrix(frame))
        #     print('\n')
        return self._get_obs()

    def step(self, action_n):
        obs_n, reward_n, done_n, info = self.env.step(action_n)
        self.frames.append(obs_n[0].copy())#antes estava só obs
        # for frame in self.frames:
        #     print(np.matrix(frame))
        #     print('\n')
        return self._get_obs(), reward_n, done_n, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis = 2)

#normalize observation space
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(1),
                                                shape=env.observation_space.shape,
                                                dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0