import gym
import numpy as np
import math
import time
import argparse
from colorama import Fore, Back, Style

from gym.envs.registration import register

from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.utils.lidarSensor import Lidar
#from indoor_explorers.render.viewer import Viewer
from multi_agent.settings import DEFAULT_CONFIG as conf
from multi_agent.indoor_explorers import IndoorExplorers


from ma_gym.wrappers import Monitor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for indoor-explorers')
    parser.add_argument('--env', default='IndoorExplorers21x21-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    env = gym.make(args.env, conf=conf)
    env = Monitor(env, directory='recordings/' + args.env, force=True)
    #env = Monitor(env, directory='recordings/'+ args.env, video_callable=lambda episode_id: True, force = True) # saves all of the episodes
    #env = Monitor(env, directory='recordings' + args.env, video_callable=lambda episode_id: episode_id%10==0) #saves the 10th episode
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()
        env.render()

        while not all(done_n):
            action_n = env.action_space.sample() #insert policy, in out case dddqn()
            print(Fore.RED) 
            print(action_n)
            print(Style.RESET_ALL)
            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            env.render()
            time.sleep(0.1)

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()