import gym
import numpy as np
from scipy.spatial import distance
import time
import pygame as pg
import argparse
import matplotlib.pyplot as plt
import json
import pickle as p
from colorama import Fore, Back, Style

from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from multi_agent.indoor_explorers import IndoorExplorers

from ma_gym.wrappers import Monitor


N_GAMES = 30
N_STEPS = 3000
DELAY = 0.
RENDER_ACTIVE = True
CONF_PATH = "/home/thedarkcurls/IndoorExplorers/non-learning/params.json"


def get_conf():

    if CONF_PATH != "":
        conf_json = json.load(open(CONF_PATH,'r'))
        conf = conf_json["env_config"]
        conf["margins"] = [3,3]
    else:
        conf["size"] = [84, 84]
        # conf["obstacles"] = 20
        # conf["lidar_range"] = 4
        # conf["obstacle_size"] = [1,3]

        conf["viewer"]["night_color"] = (0, 0, 0)
        conf["viewer"]["draw_lidar"] = True

    return conf


def find_frontiers(obs, agent_i):

    free_x, free_y = np.where(obs == 0.3)
    free_points = np.array(list(zip(free_x, free_y)))

    # diff --> temporal differences
    diff_x = [0,-1,-1,-1,0,1,1,1]
    diff_y = [1,1,0,-1,-1,-1,0,1]

    frontiers = []

    for free_x, free_y in zip(free_x, free_y):

        for dx, dy in zip(diff_x, diff_y):

            test_x = free_x + dx
            test_y = free_y + dy

            if test_x>=0 and test_x<obs.shape[0] and test_y>=0 and test_y<obs.shape[1]:
                if obs[test_x, test_y] == 0.0:
                    #add calcultate distance
                    dist = distance.cdist(env.agents[agent_i].pos, (test_x, test_y))
                    #only append the farthest
                    frontiers.append([free_x, free_y])
                    break

    return np.array(frontiers)

def find_frontiers2(obs, agent_i):
    free_x, free_y = np.where(obs == 0.3)
    free_points = np.array(list(zip(free_x, free_y)))

    # diff --> temporal differences
    diff_x = [0,-1,-1,-1,0,1,1,1]
    diff_y = [1,1,0,-1,-1,-1,0,1]
    
    frontiers = []

    #for free_x, free_y in zip(free_x, free_y):

    for dx, dy in zip(diff_x, diff_y):
        test_x = env.agents[agent_i].pos[0] + dx
        test_y = env.agents[agent_i].pos[1] + dy

        if test_x>=0 and test_x<obs.shape[0] and test_y>=0 and test_y<obs.shape[1]:
            if obs[test_x, test_y] == 0.0:
                dist = distance.cdist(env.agents[agent_i].pos, (test_x, test_y))
                



def check_collision(distances, canditate_action, obs):

    _grid_shape = env.get_grid_shape()
    for index, (x,y) in enumerate(canditate_action):

        if x<0 or y<0 or x>= _grid_shape[0] or y>= _grid_shape[1]: #outside bounds
            distances[index] = np.inf
        elif obs[x,y] > 0.3: #obstacle
            distances[index] = np.inf

    return distances


def evaluate(frontiers, obs, agent_i):
    # return the distance from each candiatate action
    canditate_action = [[env.agents[agent_i].pos[0]   , env.agents[agent_i].pos[1]+1], # action 0 is down
                        [env.agents[agent_i].pos[0]-1 , env.agents[agent_i].pos[1]],   # action 1 is left
                        [env.agents[agent_i].pos[0]   , env.agents[agent_i].pos[1]-1], # action 2 is up
                        [env.agents[agent_i].pos[0]+1 , env.agents[agent_i].pos[1]],   # action 3 is right
                        [env.agents[agent_i].pos[0]   , env.agents[agent_i].pos[1]]    # action 4 is no op
                        ]
                        
                        

    distances = distance.cdist(frontiers, canditate_action)

    evaluation = np.min(distances, axis=0)
    evaluation = check_collision(evaluation, canditate_action, obs)
    return evaluation


def get_action(obs):
    actions = []
    for agent_i in range(env.n_agents):
        frontiers = find_frontiers(env.get_full_obs(), agent_i) # obs[agent_i])
        evaluate_actions = evaluate(frontiers, env.get_full_obs(), agent_i)#)obs[agent_i], agent_i)
        actions.append(np.argmin(evaluate_actions))
    return actions


#TODO change for multi agents!
def play_game(env):

    exploration_rate = []
    # obs_n = env.reset()
    # if RENDER_ACTIVE:env.render()

    #for time_step in range(N_STEPS):
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()
        if RENDER_ACTIVE:env.render()

        while not all(done_n):
            action_n = get_action(obs_n)
            print(Fore.RED) 
            print(action_n)
            print(Style.RESET_ALL)
            obs_n, reward_n, done_n, info = env.step(action_n)
            _full_obs = env.get_full_obs()

            #TODO o que é que é mais correto? ir buscar o _full_obs diretamente do env ou o _full_obs fazer parte das obs??
            #exploration_rate.append(np.count_nonzero(obs[env.n_agents])/(obs[env.n_agents].shape[0]*obs[env.n_agents].shape[1])) #obs[env.n_agents] -> _full_obs
            exploration_rate.append(np.count_nonzero(_full_obs)/(_full_obs.shape[0]*_full_obs.shape[1]))
            #TODO isto provavelmente terá de ser mudado de sítio para fora do while, pq neste momento os agents estão done todos ao mesmo tempo 
            #(aka quando o mapa foi todo explorado ou o maximo de steps atingido, então estão sincronos)
            #mas se não forem sincronos o exploration rate não sei se deveria ficar aqui

            ep_reward += sum(reward_n)
            if RENDER_ACTIVE:env.render()
            time.sleep(DELAY)

        #------------------------------------------------------------------
        # action = get_action(obs)

        # obs, reward, done, info = env.step(action)

        # exploration_rate.append(np.count_nonzero(obs)/(obs.shape[0]*obs.shape[1]))

        # if RENDER_ACTIVE:env.render()
        # time.sleep(DELAY)

        # if done:
        #     break

    return exploration_rate


if __name__ == "__main__":
    conf = get_conf()

    #env = gym.make('mars_explorer:exploConf-v01', conf=conf)

    parser = argparse.ArgumentParser(description='Random Agent for indoor-explorers')
    parser.add_argument('--env', default='IndoorExplorers21x21-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    env = gym.make(args.env, conf=conf)
    env = Monitor(env, directory='recordings/' + args.env, force=True)

    data = []
    # for game in range(N_GAMES):
    #     print(f"Running game:{game}")
    #     # exploration_rate --> percentage of explorated area per time step (array)
    #     exploration_rate = play_game(env)
    #     data.append(exploration_rate)

    # p.dump( data, open("cost_42x42.p","wb"))


    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        print(f"Running episode:{ep_i}")
        # exploration_rate --> percentage of explorated area per time step (array)
        exploration_rate = play_game(env)
        data.append(exploration_rate)

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()
    p.dump( data, open("multi_cost_21x21.p","wb"))
