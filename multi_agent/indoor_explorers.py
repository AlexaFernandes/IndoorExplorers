import copy
import logging

import gym
import numpy as np
import math
import operator as op
from PIL import ImageColor
from colorama import Fore, Back, Style
from gym import spaces
from gym.utils import seeding
import itertools
import statistics
from statistics import mode

#import multi_agent.core
from multi_agent.utils.multi_printMaps import *
from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.utils.multi_agent_lidarSensor import Lidar

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from multi_agent.utils.draw import *


from multi_agent.settings import DEFAULT_CONFIG 

logger = logging.getLogger(__name__)


class IndoorExplorers(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, conf=DEFAULT_CONFIG):
        #assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        # assert len(agent_view_mask) == 2, 'expected a tuple of size 2 for agent view mask,' \
        #                                   ' but found {}'.format(agent_view_mask)
        #assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        # assert 0 < agent_view_mask[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        # assert 0 < agent_view_mask[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])
        self.conf=conf
        self._grid_shape = self.conf["size"] #grid_shape #tem os mesmos valores que conf["size"]
        self.n_agents = self.conf["n_agents"] 
        self.agents = self.create_agents() 
        self._max_steps = self.conf["max_steps"]
        self._step_count = None
        self._steps_beyond_done = None
        self.movementCost = self.conf["movementCost"]
        self.viewer = None
        if self.conf["check_stuck"] and (self.conf["stuck_method"] == 2) :
            self.positions = [ [] for _ in range(self.n_agents)]

        
        #definition of all the necessary matrices(maps):
        #simple map with no agents, just the walls (only map with different values for obstacles)
        self.lidar_map = None 
        # lidar map --> 1.0 obstacle
        #               0.0 free
        
        #map fully explored aka the ground truth to be compared  to -> 0.3 explored , 0.5 walls
        self.groundTruthMap = np.full(self._grid_shape, 0.0) 
        # ground truth map --> 0.5 obstacle
        #                      0.3 empty/explored

        #map wit overall view from every agent - the global map
        self._full_obs = np.full(self._grid_shape, PRE_IDS['unexplored']) #it starts totally unexplored
        # _full_obs --> 0.5 obstacle
        #               0.3 free to move
        #               0.0 unexplored
        #               >1  agent id
                                                                          

        #matrix that tells which agents are in range of each - adjacency matrix
        self.comm_range = np.full((self.n_agents,self.n_agents), 0)

        #definition of action space
        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)]) #(5 actions: L R U D NOOP) 

        #definition of the observation space: it is the view of the map in a MultiAgentObservationSpace wrappers
        # 0.0 --> unexplored
        # 0.3 --> explored/empty
        # 0.5 --> obstacle
        # >1.0 -> agents ids
        if conf["approach"] == True: #If it is the centralized approach, then we only need the _full_obs TODO check!!
            #highest value that can be observed in each cell is the 255
            self._obs_high = np.full((self._grid_shape[0], self._grid_shape[1],1), np.array(self.n_agents, dtype=np.uint8)) 
            #lowest value that can be observed in each cell is 0.0
            self._obs_low = np.full((self._grid_shape[0], self._grid_shape[1],1), np.array(0,0, dtype=np.uint8))
            self.observation_space = MultiAgentObservationSpace(
                [spaces.Box(self._obs_low, self._obs_high, (self._grid_shape[0], self._grid_shape[1],1)) for _ in range(self.n_agents+1)]) #one map for each agent  + 1 (_full_obs)
        else:
            #highest value that can be observed in each cell is the max. agent id
            self._obs_high = np.full((self._grid_shape[0], self._grid_shape[1],1), np.array(self.n_agents, dtype=np.float32)) 
            # #lowest value that can be observed in each cell is 0.0
            self._obs_low = np.full((self._grid_shape[0], self._grid_shape[1],1),np.array(0.0, dtype=np.float32))
            self.observation_space = MultiAgentObservationSpace(
                [spaces.Box(self._obs_low, self._obs_high, (self._grid_shape[0], self._grid_shape[1],1)) for _ in range(self.n_agents)]) #one map for each agent  (+ 1 (_full_obs)??)
        
        self._total_episode_reward = None
        self.seed()

    def get_grid_shape(self):
        return self._grid_shape

    def get_full_obs(self):
        return self._full_obs

    #checks if is a wall in pos
    def does_wall_exists(self, pos):
        row, col = pos
        return (self.groundTruthMap[row,col] == 0.5)
        #return PRE_IDS['wall'] in self._base_grid[row, col]

    #checks if cell is valid (aka inside bounds)
    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    #checks if it's inside bounds and there is no wall or other agent in pos
    def is_cell_spawnable(self, pos, agent_i):
        #if the pos is already take by another agent
        for i in range(agent_i): #since the spawn is done sequencially, there is no need to check the rest of the agents after agent_i
            if self.agents[i].pos == pos:
                return False        
        #otherwise if it is within bounds and no obstacle is there
        return self.is_valid(pos) and (not self.does_wall_exists(pos))

    #checks if cell is inside bounds and is vacant
    def _is_cell_vacant(self, pos):
        return (self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty/explored']))

    #check if a specific position is in line of sight of agent_i
    def is_pos_in_LOS_of(self, row, col, agent_i):
        for pos in self.lidarsIndexes[agent_i]:
            if list(pos) == [row,col]:
                return True
        return False
        
    #receives a list of agents that are in comm range that want to merge their maps
    def merge_maps(self, agent_list):
        new_merged_map = np.full(self._grid_shape, 0.0)

        for col in range(0, self._grid_shape[1]):
            for row in range(0, self._grid_shape[0]):
                for agent_i in agent_list:
                    if (self.agents[agent_i].is_alive()) and (not self.agents[agent_i].done):
                        #if that pos is in agent_i's line of sight add its info
                        if self.is_pos_in_LOS_of(row,col, agent_i):
                            #print("{} can see [{},{}]".format(agent_i,row,col))
                            new_merged_map[row][col] = self.agents[agent_i].exploredMap[row][col]
                        #save if it is a wall or explored cell
                        elif self.agents[agent_i].exploredMap[row][col] == 0.5 or self.agents[agent_i].exploredMap[row][col] == 0.3:
                            new_merged_map[row][col]= self.agents[agent_i].exploredMap[row][col]
                        #if it was an agent, check if it is still in range TODO check if this is necessary!
                        # elif self.agents[agent_i].pastExploredMap[row][col] >= 1.0:
                        #     #if it is not in range, mark the cell as explored/empty
                        #     if (self.agents[agent_i].pastExploredMap[row][col]-1) not in agent_list:
                        #         if self.conf["viewer"]["print_prompts"]:
                        #             print("{} out of range of {}".format(self.agents[agent_i].pastExploredMap[row][col]-1, agent_i))
                        #         new_merged_map[row][col] = 0.3
        
        #save the positions of all agents in range in the end, so their position isn't lost
        for agent_i in agent_list:
            if (self.agents[agent_i].is_alive()) and (not self.agents[agent_i].done):
                agent_pos = self.agents[agent_i].pos
                new_merged_map[agent_pos[0]][agent_pos[1]] = agent_i + 1

        #save a copy of the new map (it needs to be a copy otherwise they will have all the same reference and will be changing the same object)
        for agent_i in agent_list:
            if (self.agents[agent_i].is_alive()) and (not self.agents[agent_i].done):
                self.agents[agent_i].exploredMap = new_merged_map.copy()

    #updates comm_range adjacency matrix - to see which agents are in range of others
    def update_comm_range(self):
        l = []
        l.extend(range(0, self.n_agents))
        #generate a list with all the unique pairings among all agents
        combinations = list(itertools.combinations(l, 2))
        #in_range = []
        self.comm_range = np.full((self.n_agents,self.n_agents),0)

        #print(combinations)
        #check if any 2 agents are in comms range and save in the comms matrix
        for pair in combinations:
            if (self.agents[pair[0]].is_alive() and (not self.agents[pair[0]].done) ) and ((not self.agents[pair[1]].done) and (self.agents[pair[1]].is_alive())) and (self.agents[pair[0]].in_range(self.agents[pair[1]]) == True):
                #print(pair)
                #in_range.append(pair)
                self.comm_range[pair[0]][pair[1]] = self.comm_range[pair[1]][pair[0]] = 1 
            else:
                self.comm_range[pair[0]][pair[1]] = self.comm_range[pair[1]][pair[0]] = 0

    #auxiliary function - DFS graph search to check which agents are in range of others
    def DFSUtil(self, temp, agent_i, visited):
 
        # Mark the current vertex as visited
        visited[agent_i] = True

        # Store the vertex to list
        temp.append(agent_i)

        # Repeat for all vertices adjacent
        # to this vertex v
        for j in range(0, self.n_agents):
            if (self.agents[agent_i].is_alive() and not self.agents[agent_i].done) and (self.agents[j].is_alive() and not self.agents[j].done): #if both involved agents are alive and not done yet
                if self.comm_range[agent_i][j] == 1: #if they are connected
                    if visited[j] == False:
                        # Update the list
                        temp = self.DFSUtil(temp, j, visited)
                    
        return temp

    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.n_agents):
            if (not self.agents[i].is_alive()) or self.agents[i].done: #if the agent is not alive or it is done
                visited.append(True)
            else:
                visited.append(False)
        for agent_i in range(self.n_agents):
            if (self.agents[agent_i].is_alive() and not self.agents[agent_i].done) and (visited[agent_i] == False): #maybe it is not necessary to do that double check, bc it is already marked as bisited above
                temp = []
                cc.append(self.DFSUtil(temp, agent_i, visited))
        return cc

    def check_who_in_comm_range_of(agent_i, self):
        l = []
        l.extend(range(0, self.n_agents))
        l.remove(agent_i)

        #check for collisions between agent_i and agent_x
        for agent_x in l:
            if self.agents[agent_i].in_range(self.agents[agent_x]) == True:
                in_range.append(agent_x)
            
        #return list of agents idxs in range of agent_i
        return in_range

    #is it necessary??
    def check_for_collision(self):
        #generate a list with all the unique pairings among all agents
        combinations = list(itertools.combinations(self.n_agents, 2))

        #check for collisions between any 2 agents
        for pair in combinations:
            if self.agents[pair[0]].pos == self.agents[pair[1]].pos:
                self.agents[pair[0]].collision=True
                self.agents[pair[1]].collision=True
                collisions.append(pair)

        return collisions

    #create a random map and subconsequent maps (such as lidar_map)
    def __create_grid(self):
        gen = Generator(self.conf) #generates map with 1's and 0's
        randomMap = gen.get_map().astype(np.double)

        #save lidar map
        # lidar map --> 1.0 obstacle
        #               0.0 free
        self.lidar_map = randomMap.copy() #save a map not explored, with 0.0 (unexplored) and 1.0 (walls)

        #correct values for groundTruthMap:
        # ground truth map --> 0.5 obstacle
        #                      0.3 empty/explored
        randomMap = randomMap.copy() #save the ground truth, fully explored
        randomMap[randomMap == 1.0] = PRE_IDS['wall'] #obstacle value is 0.5
        randomMap[randomMap == 0.0] = PRE_IDS['empty/explored'] #explored cells, which value is 0.3
        
        return randomMap

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=UNEXPLORED_COLOR)

    #updates the agent's info (lidar scans, maps and comm_range adjacency matrix)
    def __update_agents_view(self):
        # initialing position is explored
        self._activateLidars()
        # for agent_i in range(self.n_agents):
        #     for pos in self.lidarsIndexes[agent_i]:
        #         print("{} can see {}".format(agent_i, pos))

        #update every agent's explored map and _full_obs
        self._updateMaps()
        #printAgentsMaps(self.agents,self.n_agents)

        #update comms matrix
        self.update_comm_range()

    #if an action is valid (inside bounds and no colision) then it is applied, otherwise the agent does not move
    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agents[agent_i].pos) #agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = [curr_pos[0], curr_pos[1]]
        else:
            raise Exception('Action Not found!')

        if curr_pos == None or next_pos == None:
            Exception("whaaat?")

         #since they move one at a time there is no risk of collision between agents
        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agents[agent_i].pos = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty/explored']
            if self.conf["viewer"]["print_prompts"]: print("agent {} - alright".format(agent_i))

        elif next_pos in self.obstacles_idx: #collision with an obstacle ( I am adding collisions so it does not get stuck against walls)
            self.agents[agent_i].collision = True
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty/explored'] #it disappears from the map
            if self.conf["viewer"]["print_prompts"]: print("agent {} - obstacle".format(agent_i))

        elif not self.is_valid(next_pos): # is outside of bounds
            self.agents[agent_i].out_of_bounds = True
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty/explored'] #it disappears from the map
            if self.conf["viewer"]["print_prompts"]: print("agent {} - outside".format(agent_i))

        else: #if a random action is no op or possible collision with other agent -> does nothing (keeps the same pos)
            self.agents[agent_i].pos = curr_pos
            if self.conf["viewer"]["print_prompts"]: print("agent {} - no op or prevented agent collision".format(agent_i))

        self.agents[agent_i].pastExploredMap = self.agents[agent_i].exploredMap.copy()
        self.__update_agents_view() 

    #check which would the the next_pos if hipotethicaly the specified "move" was applied
    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    #create instances of lidars for each agent
    def create_lidars(self):
        # for lidar --> 0 free cell
        #               1 obstacle
        #create an array of lidar, one per agent
        self.ldr = [Lidar(r=self.conf["lidar_range"],
                         channels=self.conf["lidar_channels"],
                         _map=self.lidar_map)                   for _ in range(self.n_agents)]
        
        #create list of obstacle indexes
        obstacles_idx = np.where(self.groundTruthMap == 0.5)
        obstacles_x = obstacles_idx[0]
        obstacles_y = obstacles_idx[1]
        self.obstacles_idx = np.stack((obstacles_x, obstacles_y), axis=1)
        self.obstacles_idx = [list(i) for i in self.obstacles_idx]

    #update each agent's map and _full_obs with new info gathered from lidar scans
    def _updateMaps(self):
        #update maps with lidar info
        for agent_i in range(self.n_agents):
            if self.agents[agent_i].is_alive() and (not self.agents[agent_i].done): #if agent_i is alive and isn't done

                lidarX = self.lidarsIndexes[agent_i][:,0]
                lidarY = self.lidarsIndexes[agent_i][:,1]

                #update what lidars has scanned without any agents
                self.agents[agent_i].exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY] 
                self._full_obs[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]        
        

        #update agents pos in _full_obs: -> needs to be done after, so the lidar info does not override the pos of agents in _full_obs
        for agent_i in range(self.n_agents):
            if self.agents[agent_i].is_alive() and (not self.agents[agent_i].done): #if agent_i is alive and isn't done
                #self.agents[agent_i].exploredMap[self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]] = (agent_i + 1) 
                self._full_obs[self.agents[agent_i].pos[0]][self.agents[agent_i].pos[1]] = (agent_i + 1) #note that here we sum 1
                #print(np.where(self._full_obs==(agent_i+1)))
        # print("full_obs after update")
        # printMap(self._full_obs, self.n_agents)
        #print("both")
        #print2Map(self.agents[0].exploredMap, self.n_agents,self.agents[1].exploredMap)

        # #other agent's pos who are in line of sight must be updated last
        # for agent_i in range(self.n_agents):
        #     if self.agents[agent_i].is_alive() and (not self.agents[agent_i].done):
        #         lidarX = self.lidarsIndexes[agent_i][:,0]
        #         lidarY = self.lidarsIndexes[agent_i][:,1]
        #         self.agents[agent_i].exploredMap[lidarX, lidarY] = self._full_obs[lidarX, lidarY] 
        #         #self.agents[agent_i].exploredMap[self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]] = (agent_i + 1) 

        #update each agent's map with _full_obs info and verify if each agent only has 1 entry for any other agent visible
        for agent_i in range(self.n_agents):
            if self.agents[agent_i].is_alive() and (not self.agents[agent_i].done):
                #update other agents' pos
                lidarX = self.lidarsIndexes[agent_i][:,0]
                lidarY = self.lidarsIndexes[agent_i][:,1]
                self.agents[agent_i].exploredMap[lidarX, lidarY] = self._full_obs[lidarX, lidarY] 

                #verify if for each agent's pos in agent_i's map there is only 1 entry of every other agent (agent_j)
                #and if that entry is correct
                for agent_j in range(self.n_agents): 
                    if agent_i != agent_j: #their own pos must be already correct, so it is not necessary to verify
                        if self.agents[agent_j].is_alive() and (not self.agents[agent_j].done):
                            #get coordinates of each position of agent_j in the current map
                            x,y = np.where(self.agents[agent_i].exploredMap==agent_j+1)
                            coord = np.array(list(zip(x, y))) #is an array with pairs of coordinates of agent's positions

                            #print("coord:{} of agent {} in agent's {} map".format(coord, agent_j, agent_i))
                            if len(coord) == 0:
                                #print("empty")
                                continue
                            elif len(coord) == 1:
                                #check if corresponding pos is right and agent_j is still in LOS of agent_i
                                pos=coord[0]
                                if list(pos) == self.agents[agent_j].pos:
                                    if self.is_pos_in_LOS_of(pos[0],pos[1],agent_i):
                                        #print("alright")
                                        continue
                                    else: #remove entry 
                                        #print("out of sight, removed")
                                        self.agents[agent_i].exploredMap[pos[0]][pos[1]]=PRE_IDS["empty/explored"] #mark as explored aka 0.3
                                else:
                                    #print("u have the wrong pos, fix it!")
                                    self.agents[agent_i].exploredMap[pos[0]][pos[1]]=PRE_IDS["empty/explored"] #mark as explored aka 0.3
                                    self.agents[agent_i].exploredMap[self.agents[agent_j].pos[0]][self.agents[agent_j].pos[1]]= agent_j + 1
                                # TODO ALTERNATIVE CODE, NOT TESTED: 
                                # if list(pos) == self.agents[agent_j].pos:
                                #      print("alright")
                                #      continue
                                # else: #READ HEREEE! i think this "else" in the code above is impossible (aka having a wrong entry in the explored map of any agent), so instead of having to run self.is_pos_in_LOS_of before, which adds a little bit o run time, I would like to test without it and just remove the wrong entry -> this is just a  theory, so I will leave this version commented 
                                #     print("u have the wrong pos, fix it!")
                                #     self.agents[agent_i].exploredMap[pos[0]][pos[1]]=PRE_IDS["empty/explored"] #mark as explored aka 0.3
                            else:
                                #in case there is more than 1 entry of agent_j
                                for pos in coord:
                                    #compare with the correct pos of agent_j and check that agent_j is still in LOS of agent_i
                                    if list(pos) == self.agents[agent_j].pos and self.is_pos_in_LOS_of(pos[0],pos[1],agent_i):
                                        #print("this is the right pos {}".format(pos))
                                        continue
                                    else:
                                        #mark the rest of the entries as explored  
                                        self.agents[agent_i].exploredMap[pos[0]][pos[1]]=PRE_IDS["empty/explored"] #mark as explored aka 0.3
                        else:
                            #in case agent_j is not alive or done
                            #then remove any entries of that agent in agent_i's map    
                            x,y = np.where(self.agents[agent_i].exploredMap==agent_j+1)    
                            coord = np.array(list(zip(x, y)))
                            for pos in coord:
                                #print("agent {} is gone".format(agent_j))
                                self.agents[agent_i].exploredMap[pos[0]][pos[1]]=PRE_IDS["empty/explored"] #mark as explored aka 0.3

        # print("full_obs")
        # printMap(self._full_obs, self.n_agents)
        # print("both-after all")
        # print2Map(self.agents[0].exploredMap, self.n_agents,self.agents[1].exploredMap)

    #activates the lidar of each agent
    def _activateLidars(self):

        indexes = [None for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents): 
            if self.agents[agent_i].is_alive() and (not self.agents[agent_i].done): #if agent_i is alive and isn't done
                self.ldr[agent_i].update(self.agents[agent_i].pos)
                indexes[agent_i] = self.ldr[agent_i].idx

        self.lidarsIndexes = indexes

    #initialization of the map and every agent's position
    def __init_full_obs(self):
        #creates new map
        self.groundTruthMap = self.__create_grid()
        self._full_obs = np.full(self._grid_shape, PRE_IDS['unexplored'])

        self.create_lidars()

        # create an empty exploredMap for each agent and
        # inserts agents at random locations 
        for agent_i in range(self.n_agents):
            # 0 if not visible/visited, 1 if visible/visited
            self.agents[agent_i].exploredMap = np.zeros(self._grid_shape, dtype=np.double) #exploredMaps[agent_i] = np.zeros(self.SIZE, dtype=np.double)

            if self.conf["random_spawn"]: random_spawn=True
            else: random_spawn = False

            while True:
                if random_spawn:
                    pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                           self.np_random.randint(0, self._grid_shape[1] - 1)]
                else:
                    pos = self.conf["initial"][agent_i]

                if self.is_cell_spawnable(pos, agent_i):
                    self.agents[agent_i].pos = pos 
                    break
                else:
                    #set flag to generate a new possible spawn position
                    random_spawn=True
            #reset spawn flag
            random_spawn=self.conf["random_spawn"]
        
        #this replaces the two commented lines (activateLidar + _updateMaps + update_comms)
        self.__update_agents_view()

        #create grid for later render
        self.__draw_base_img()

    #reset function
    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.reset_agents() 
        if self.conf["check_stuck"] and (self.conf["stuck_method"] == 2) :
            self.positions = [ [] for _ in range(self.n_agents)]
        self.comm_range = np.full((self.n_agents,self.n_agents), 0)

        #criar novo mapa e dá spawn de novo dos agents
        #cria lidares novos (isto inclui criar novos obstaulos) com base no novo mapa
        #activa os lidars e dá update dos explored maps de cada um 
        #com base no que cada um consegue ver e na matrix de comms
        self.__init_full_obs()
        print(self.comm_range)
        #print("reset-before merge")
        #printAgentsMaps(self.agents, self.n_agents)
        #print2Map(self.agents[0].exploredMap,self.n_agents,self.agents[1].exploredMap)
        if self.n_agents > 1:
            groups_in_range = []
            groups_in_range = self.connectedComponents()
            if self.conf["viewer"]["print_prompts"]:
                print("reset, groups in range: {}".format(groups_in_range))
            for group in groups_in_range:
                if len(group)>1:
                    self.merge_maps(group)
        #print("reset-after merge_maps")
        #printAgentsMaps(self.agents,self.n_agents)
        #reset other vars
        self._step_count = 0
        self._steps_beyond_done = None
        
        if self.conf["viewer"]["print_prompts"]:
            print("reset done")
        return self.get_agents_obs()

    #step function
    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        rewards = np.zeros(self.n_agents)
        self._step_count += 1
        if self.conf["viewer"]["print_prompts"]:
            print("Start of step {}".format(self._step_count))

        #randomize the order of action taking, so there is no hierarchy between agents
        #it randomizes the order of the agents' indeces
        idxs = np.arange(0,len(agents_action))
        action_dict = dict(enumerate(agents_action)) #save in a dict to preserve original pairs fo key-value (agent_id - corresponding action)
        np.random.shuffle(idxs) #shufle the indeces

        #apply chosen action
        for i in range(len(agents_action)):
            if not (self.agents[idxs[i]].done): 
                if self.conf["check_stuck"] and (self.conf["stuck_method"]==2):
                    #save the history of 50 positions for later check
                    if len(self.positions[idxs[i]]) < 50 :
                        self.positions[idxs[i]].append(tuple(self.agents[idxs[i]].pos)) #pos needs to be converted to a tuple, so the list is hashable
                    else: #once it gets to 50 pop the 1st/oldest elem and append the most recent one at the end
                        self.positions[idxs[i]].pop(0)
                        self.positions[idxs[i]].append(tuple(self.agents[idxs[i]].pos))
                #print("Start of agent {}{} action {}-{}{}".format(TEXT_AGENT_COLORS[idxs[i]], idxs[i], action_dict[idxs[i]],ACTION_MEANING[action_dict[idxs[i]]], TEXT_AGENT_COLORS[idxs[i]] ))
                self.__update_agent_pos(idxs[i], action_dict[idxs[i]])
                #print("After agent's {} action done:".format(idxs[i]))
                
                #print("{}{} {}-{}{}".format(TEXT_AGENT_COLORS[idxs[i]], idxs[i], action_dict[idxs[i]],ACTION_MEANING[action_dict[idxs[i]]], TEXT_AGENT_COLORS[idxs[i]] ))
                #print("{} {} {}".format(TEXT_AGENT_COLORS[idxs[i]],self.agents[idxs[i]].pos,TEXT_AGENT_COLORS[idxs[i]]))
                #print(Style.RESET_ALL)
        #print("step-before merge")
        #printAgentsMaps(self.agents, self.n_agents)

        #communicate maps
        #check if they are in comm range and then change info
        if self.n_agents > 1:
            groups_in_range = []
            groups_in_range = self.connectedComponents()
            if self.conf["viewer"]["print_prompts"]:
                print("after actions, groups in range: {}".format(groups_in_range))
            for group in groups_in_range:
                if len(group)>1:
                    self.merge_maps(group)  
        #print("step - after merge")
        #check if agents are done and compute rewards
        for agent_i in range(self.n_agents):
            self._checkDone(agent_i)
            rewards[agent_i] = self.agents[agent_i].reward
            self._total_episode_reward[agent_i] += rewards[agent_i] #save cumulative reward of the episode for each agent

        if np.count_nonzero(self._full_obs) > self.conf["percentage_explored"]*(self._grid_shape[0]*self._grid_shape[1]): #if _full_obs is the defined % explored
            #then end episode, set all dones to true
            for agent_i in range(self.n_agents):
                self.agents[agent_i].done = True 
                #if self.agents[agent_i].is_alive():
                    #self._full_obs[self.agents[agent_i].pos[0]][self.agents[agent_i].pos[1]] = PRE_IDS['empty/explored'] #for correct rendering, does not affect calculation of reward
                #fill_cell(self._base_img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='white') #for correct redering


        # Check for episode overflow
        if all(self.get_agents_dones()):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1


        return self.get_agents_obs(), rewards, self.get_agents_dones(), {'total_episode_reward' : self._total_episode_reward} 

    #check specific conditions for each agent
    def _checkDone(self, agent_i):
        #if the agent is already done, set reward to 0
        if self.agents[agent_i].done : 
            self.agents[agent_i].reward = 0
            return
            
        #if max_steps has been reached end episode
        if (self._step_count >= self._max_steps): #and (not self.agents[agent_i].done):
            self.agents[agent_i].done = True
            #if self.agents[agent_i].is_alive():
                #print(self.agents[agent_i].pos)
                #self._full_obs[self.agents[agent_i].pos[0]][self.agents[agent_i].pos[1]] = PRE_IDS['empty/explored'] #for correct rendering, does not affect calculation of reward
            #fill_cell(self._base_img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='white')

        #Calculation of rewards:
        #check if any agent has explored the defined % of the map, if so task is complete, give extra reward and end episode (this last part is done after)
        if np.count_nonzero(self.agents[agent_i].exploredMap) > self.conf["percentage_explored"]*(self._grid_shape[0]*self._grid_shape[1]):
            if self.agents[agent_i].is_alive(): fill_cell(self._base_img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='white')
            self.agents[agent_i].done = True
            self.agents[agent_i].reward = self.conf["bonus_reward"]
        elif self.agents[agent_i].collision: #if it crashes against a wall
            self.agents[agent_i].done = True
            self.agents[agent_i].reward = self.conf["collision_reward"]
        elif self.agents[agent_i].out_of_bounds: #if it goes out of the bounds
            self.agents[agent_i].done = True
            self.agents[agent_i].reward = self.conf["out_of_bounds_reward"]        
        else:
            pastExploredCells = np.count_nonzero(self.agents[agent_i].pastExploredMap)
            currentExploredCells = np.count_nonzero(self.agents[agent_i].exploredMap)

            if self.conf["check_stuck"]:
                #for stuck method 1:
                if self.conf["stuck_method"] == 1:
                    #update flag
                    if(currentExploredCells - pastExploredCells)==0: #if it does not learn something new
                        self.agents[agent_i].stuck += 1 #increment flag to know if it is stuck
                        #print(self.agents[agent_i].stuck)

                        #check if stuck condition has been met
                        if self.agents[agent_i].stuck >= math.ceil(self._grid_shape[0]*self._grid_shape[1]):#(2*math.sqrt(self._grid_shape[0]**2 + self._grid_shape[1]**2)):
                            self.agents[agent_i].done = True
                            print("GOT STUCK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            self.agents[agent_i].reward = self.conf["stuck_reward"]
                            return
                    else: #if it discovers at least 1 new cell, then reset the flag
                        self.agents[agent_i].stuck = 0

                #for stuck method 2:
                elif self.conf["stuck_method"] == 2:
                    #update flag
                    lst = self.positions[agent_i]
                    #the mode of self.positions[agent_i] corresponds to the most frequent element(pos) of the list 
                    #getting the countOf, gives the number of times it appears on the list
                    if op.countOf(lst, mode(lst)) >= 20 : #so it the most frequent pos appears more than 10 times in 50 movements
                        self.agents[agent_i].stuck = 1 #aka True
                    else:
                        self.agents[agent_i].stuck = 0

                    #check if stuck condition has been met
                    if self.agents[agent_i].stuck == 1:
                        self.agents[agent_i].done = True
                        print("GOT STUCK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!222222222222222222222222")
                        self.agents[agent_i].reward = self.conf["stuck_reward"]
                        return
                else:
                    logger.warn(
                        "You must define a stuck_method check, either 1 or 2"
                    )
           
            
            print("new cells explored: {}".format((currentExploredCells-pastExploredCells)))
            self.agents[agent_i].reward = self.conf["new_cell_disc_reward"]*(currentExploredCells - pastExploredCells) - self.movementCost
            print("agent reward:{}".format(self.agents[agent_i].reward))

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        img = copy.copy(self._base_img)            

        #render updated map
        for col in range(0, self._grid_shape[1]):
            for row in range(0, self._grid_shape[0]):
                pos = [row,col]
                if self._full_obs[row][col] == PRE_IDS['empty/explored']: #0.3 #explored cells
                    fill_cell(img,pos, cell_size=CELL_SIZE, fill='white')
                    draw_cell_outline(img, pos, cell_size=CELL_SIZE, fill='black',width=1)
                if self._full_obs[row][col] == PRE_IDS['wall']: #0.5: #walls
                    fill_cell(img,pos, cell_size=CELL_SIZE, fill=WALL_COLOR)

        #render lidar
        for agent_i in range(self.n_agents):
            #if not self.agents[agent_i].done:
            if self.agents[agent_i].is_alive():
                #draw lidar fov (if the agent is done, then dont draw lidar fov -> to visually see that it is the end and lidars are turned off)
                if self.conf["viewer"]["draw_lidar"] == True and (not self.agents[agent_i].done):
                    for neighbour in self.lidarsIndexes[agent_i]:
                        if not self._full_obs[neighbour[0]][neighbour[1]] == PRE_IDS['wall']:
                            fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
                    fill_cell(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
                else:
                    fill_cell(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='white')
                draw_cell_outline(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='black',width=1)
        
        #render agents
        for agent_i in range(self.n_agents):
            #if not self.agents[agent_i].done:
            if self.agents[agent_i].is_alive(): #always render agents, except when agents collide or are out of bounds
                draw_circle(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=self.agents[agent_i].color)
                write_cell_text(img, text=str(agent_i), pos=self.agents[agent_i].pos, cell_size=CELL_SIZE,
                                fill='white', margin=0.4)  
                draw_square_outline(img, self.agents[agent_i].pos, self.agents[agent_i].c_range, cell_size=CELL_SIZE, fill=self.agents[agent_i].color, width = 2)
            
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            #este print tem que ser depois de atualizar o viewer pq senão acontece aquele comportamento estranho em que o viewer parece estar atrasado
            #mas simplesmente não foi renderizado depois de ter os dados atualizados
            if self.conf["viewer"]["print_map"] == True:
                #print("render")
                #printMap(self._full_obs, self.n_agents)
                #print2Map(self.agents[0].exploredMap, 1, self.agents[0].pastExploredMap)
                printAgentsMaps(self.agents, self.n_agents)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def create_agents(self):
        agents = [Agent() for i in range(self.n_agents)]
        for i, agent in enumerate(agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.color = AGENT_COLORS[i]
            
        return agents

    def reset_agents(self):
        for agent_i in range(self.n_agents):
            self.agents[agent_i].done = False
            self.agents[agent_i].reward = 0
            self.agents[agent_i].pos = None
            self.agents[agent_i].stuck = 0
            self.agents[agent_i].collision = False
            self.agents[agent_i].out_of_bounds = False 
            self.agents[agent_i].c_range = self.conf["comm_range"]
    
    #
    def get_agents_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            _agent_i_obs = self.agents[agent_i].exploredMap 
            _agent_i_obs = np.reshape(_agent_i_obs, (self._grid_shape[0], self._grid_shape[1],1))

            _obs.append(_agent_i_obs)
        if self.conf["approach"] == True:
            _obs.append( np.reshape(self._full_obs, (self._grid_shape[0], self._grid_shape[1],1) ) )

        return _obs

    def get_agents_dones(self):
        return [self.agents[agent_i].done for agent_i in range(self.n_agents)]



class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.name = ''
        self.id = None
        self.pos = None #position
        self.reward = 0 
        self.done = False
        #each agent has its own explored map
        self.exploredMap = []
        self.pastExploredMap = []
        #flag to signal stuck status
        self.stuck = 0
        #flag for collisions
        self.collision = False
        #flag for out of bounds status
        self.out_of_bounds = False
        #communication range
        self.c_range = 3.0
        # communication noise amount
        self.c_noise = None #not implemented
        # color
        self.color = None
        # state
        self.state = None
        # action
        self.action = None

    #it's considered in range, inside a square with distance of c_range squares around the agent
    def in_range(self, agent2):
        delta_pos = abs(np.subtract(np.array(self.pos), np.array(agent2.pos)) ) 
        if delta_pos[0] > self.c_range*2 or delta_pos[1] > self.c_range*2 :
            return False
        else:
            return True

    #checks if the agent has not collided or out of bounds
    def is_alive(self):
        return (not self.collision) and (not self.out_of_bounds) 


AGENT_COLORS = [(30, 150, 245),(220, 10, 10),(0, 204, 0),(255, 215, 0)]
TEXT_AGENT_COLORS = ["\u001b[34m ", "\033[91m", "\u001b[32m", "\u001b[33m"]
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
UNEXPLORED_COLOR = ImageColor.getcolor('lightgrey', mode='RGB')

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'wall': 0.5,
    'unexplored': 0.0,
    'empty/explored': 0.3
}
