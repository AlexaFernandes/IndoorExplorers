import copy
import logging

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding
import itertools

import multi_agent.core
from multi_agent.utils.multi_printMaps import *
from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.utils.multi_agent_lidarSensor import Lidar

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from multi_agent.utils.draw import *


from multi_agent.settings import DEFAULT_CONFIG 

logger = logging.getLogger(__name__)

RANDOM_SPAWN=False #TODO colocar no ficheiro depois

class IndoorExplorers(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, conf=DEFAULT_CONFIG,# grid_shape=(21, 21),# n_agents=4,
                 full_observable=False, penalty=-0.5, step_cost=-0.01):#, prey_capture_reward=5 ): #, max_steps=100,
                 #agent_view_mask=(5, 5)):
        #assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        # assert len(agent_view_mask) == 2, 'expected a tuple of size 2 for agent view mask,' \
        #                                   ' but found {}'.format(agent_view_mask)
        #assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        # assert 0 < agent_view_mask[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        # assert 0 < agent_view_mask[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])
        self.conf=conf
        self._grid_shape = conf["size"] #grid_shape #tem os mesmos valores que conf["size"]
        self.n_agents = conf["n_agents"] 
        self.agents = self.create_agents() #TODO verificar se é preciso inicializar mais alguma coisa nesta função
        self._max_steps = conf["max_steps"]
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._step_cost = step_cost
        #self._prey_capture_reward = prey_capture_reward
        #self._agent_view_mask = agent_view_mask

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)]) #por agora vou deixar 5 ações (L R U D NOOP) TODO: add share info
        #self.agent_pos = {_: None for _ in range(self.n_agents)} 
        #self.exploredMaps = [None for _ in range(self.n_agents)] #explored map of every agent
        #self._agent_dones = [False for _ in range(self.n_agents)] #this is done when creating the agents

        self.lidar_map = None # with no agents, just the walls
                              # lidar map --> 1.0 obstacle
                              #               0.0 free
        self.groundTruthMap = self.__create_grid()  #map fully explored aka the ground truth to be compared  to -> 0.3 explored , 0.5 walls
                                                    # ground truth map --> 0.5 obstacle
                                                    #                      0.3 empty/explored
        self._full_obs = np.full(self._grid_shape, PRE_IDS['unexplored']) #map wit overall view from every agent - the global map
                                                                          #it starts totally unexplored
                                                                          # _full_obs --> 0.5 obstacle
                                                                          #               0.3 free to move
                                                                          #               0.0 unexplored
                                                                          #               >1  agent id
        

        self.viewer = None
        self.full_observable = full_observable

        #observation space is the view of the map
        # 0.0 --> unexplored
        # 0.3 --> explored/empty
        # 0.5 --> obstacle
        # >1.0 -> agents ids
        #highest value that can be observed in each cell is the max. agent id
        self._obs_high = np.full(grid_shape, np.array(self.n_agents, dtype=np.float32)) 
        #lowest value that can be observed in each cell is 0.0
        self._obs_low = np.full(grid_shape,np.array(0.0, dtype=np.float32))
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high, grid_shape) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()


    #checks if is a wall in pos
    def does_wall_exists(self, pos):
        row, col = pos
        return (self.groundTruthMap[row,col] == 1.0)
        #return PRE_IDS['wall'] in self._base_grid[row, col]

    #checks if cell is valid (aka inside bounds)
    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    #checks if it's inside bounds and there is no wall in pos
    def is_cell_spawnable(self, pos):
        return self.is_valid(pos) and (not self.does_wall_exists(pos))

    #checks if cell is inside bounds and is vacant
    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty/explored'])

    #receives a list of agents that are in comm range that want to merge their maps
    def merge_maps(self, agent_list):
        new_merged_map = np.array.full(self._grid_shape, 0.0)

        for col in range(0, self._grid_shape[1]):
            for row in range(0, self._grid_shape[0]):
                for agent_i in agent_list:
                    if self.agents[agent_i].exploredMap[row][col] != 0.0:
                        new_merged_map[row][col]= self.agents[agent_i].exploredMap[row][col]

        #save a copy of the new map (it needs to be a copy otherwise they will have all the same reference and will be changing the same object)
        for agent_i in agent_list:
            self.agents[agent_i].exploredMap = new_merged_map.copy()
        # if (self.exploredMap==map).all():
        #     return
        # else:  

    #se calhar faria mais sentido a partir de um certo agent quem está in range
    def check_who_in_comm_range(self):
        l = []
        l.extend(range(0, self.n_agents))
        #generate a list with all the unique pairings among all agents
        combinations = list(itertools.combinations(l, 2))

        #check for collisions between any 2 agents
        for pair in combinations:
            if self.agents[pair[0]].in_range(self.agents[pair[1]]) == True:
                in_range.append(pair)
            
        #return list of pairs of agents in range of eachother
        return in_range

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

    #TODO when 2 agents are in risk of collision communicate who does what
    def communicate_move(self, agent1, agent2):
        #it is defined that agent1 takes the lead and choses first


        #then agent2 makes their decision accordingly

        return    

    def __create_grid(self):
        gen = Generator(self.conf)
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

    #função que atualiza a info de cada agente 
    def __update_agents_view(self):
        # initialing position is explored
        self._activateLidars()

        #update every agent's explored map and _full_obs
        self._updateMaps()


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
            pass
        else:
            raise Exception('Action Not found!')

         #since they move one at a time there is no risk of collision
        if next_pos is not None and self._is_cell_vacant(next_pos):
                self.agents[agent_i].pos = next_pos #agent_pos[agent_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty/explored']
                self.__update_agents_view()

        #WHAT DOES IT DO IF a random action is not valid ?? -> does nothing??
            # else: # collision risk -> choose a different action
            #     self.action_space.sample() #TODO MUDAR ISSTOOOO!! não sei bem como fazer, mas tem de decidir de outra forma


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

    #CONFIRM!!
    def create_lidars(self):
        # for lidar --> 0 free cell
        #               1 obstacle
        #create an array of lidar, one per agent
        self.ldr = [Lidar(r=self.conf["lidar_range"],
                         channels=self.conf["lidar_channels"],
                         map=self.lidar_map)                     for _ in range(self.n_agents)]
        
        #create list of obstacle indexes
        obstacles_idx = np.where(self.groundTruthMap == 1.0)
        obstacles_x = obstacles_idx[0]
        obstacles_y = obstacles_idx[1]
        self.obstacles_idx = np.stack((obstacles_x, obstacles_y), axis=1)
        self.obstacles_idx = [list(i) for i in self.obstacles_idx]

    #CONFIRM!!
    def _updateMaps(self):

        for agent_i in range(self.n_agents):
            self.agents[agent_i].pastExploredMaps = self.agents[agent_i].exploredMap.copy()

            lidarX= self.lidarsIndexes[agent_i][:,0]
            lidarY = self.lidarsIndexes[agent_i][:,1]

            #update what lidars has scanned
            self.agents[agent_i].exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY] 
            self._full_obs[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]

            #update agents pos:
            self.agents[agent_i].exploredMap[self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]] = agent_i + 1 
            self._full_obs[self.agents[agent_i].pos[0]][self.agents[agent_i].pos[1]] = agent_i + 1 #note that here we sum 1

    #CONFIRM!!
    def _activateLidars(self):

        indexes = [None for _ in range(self.n_agents)]

        for agent_i in range(self.n_agents): 
            self.ldr[agent_i].update(self.agents[agent_i].pos) #TODO CONFIRMAR!! alternativa: [self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]]
            #thetas[agent_i], ranges[agent_i] = self.ldr[agent_i].thetas, self.ldr[agent_i].ranges
            indexes[agent_i] = self.ldr[agent_i].idx

        self.lidarsIndexes = indexes


    #CONFIRMAR!! esta vai ser a função de inicialização do mapa e das posições de cada agente
    def __init_full_obs(self):
        #creates new map
        self.groundTruthMap = self.__create_grid()
        self._full_obs = np.full(self._grid_shape, PRE_IDS['unexplored'])

        self.create_lidars()

        # create an empty exploredMap for each agent and
        # inserts agents at random locations #TODO mudar para que dê spawn nos cantos do mapa
        for agent_i in range(self.n_agents):
            # 0 if not visible/visited, 1 if visible/visited
            self.agents[agent_i].exploredMap = np.zeros(self._grid_shape, dtype=np.double) #exploredMaps[agent_i] = np.zeros(self.SIZE, dtype=np.double)

            if RANDOM_SPAWN: random_spawn=True
            else: random_spawn = False

            while True:
                if random_spawn:
                    pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                           self.np_random.randint(0, self._grid_shape[1] - 1)]
                else:
                    pos = self.conf["initial"][agent_i]

                if self.is_cell_spawnable(pos):
                    self.agents[agent_i].pos = pos #agent_pos[agent_i] = pos
                    break
                else:
                    #set flag to generate a new possible spawn position
                    random_spawn=True
            #reset spawn flag
            random_spawn=RANDOM_SPAWN
        
        #this replaces the two commented lines (activateLidar + _updateMaps)
        self.__update_agents_view()
        # # initial position is explored
        # self._activateLidars()

        # #update everyone's explored maps
        # self._updateMaps()

        #create grid for later render
        self.__draw_base_img()


    #CONFIRM!!
    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        #old: self.agent_pos = {} -> done in reset_agents() below
        self.reset_agents() 

        #criar novo mapa e dá spawn de novo dos agents
        #cria lidares novos (isto inclui criar novos obstaulos) com base no novo mapa
        #activa os lidars e dá update dos explored maps de cada um 
        #com base no que cada um consegue ver
        self.__init_full_obs()


        #repor todas as outras variaveis
        self._step_count = 0
        self._steps_beyond_done = None
        #old: _agent_dones = [False for _ in range(self.n_agents)] -> done in reset_agents a few lines up

        return self.get_agents_obs()

    #CONFIRMAR!!
    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        
        #apply chosen action
        for agent_i, action in enumerate(agents_action):
            if not (self.agents[agent_i].done): #_agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
        
        #compute rewards for each agent 
        rewards = [self._computeReward(agent_i) for agent_i in range(self.n_agents)]


        #check if max steps as been reached or the all map as been explored
        if (self._step_count >= self._max_steps) or (0.0 not in self._full_obs):  #old: or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self.agents[i].done = True  #_agent_dones[i] = True


        #TODO CHECK IF DONE:
        #if _checkDone():


        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

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

        #TODO colocar na função render()?
        # for agent in range(self.n_agents):
        #     printMap(self.agents[agent_i].exploredMap)

        return self.get_agents_obs(), rewards, self.get_agents_dones(), {'other info' : None}  #_agent_dones
        #the info parameter was: {'prey_alive': self._prey_alive} in the original code, TODO see what extra indo would be useful for me

    #TODO: this is from the other code, not from here, delete later
    def _checkDone(self):

        if self.timeStep > self.maxSteps:
            self.done = True
        elif np.count_nonzero(self.exploredMap) > 0.99*(self.SIZE[0]*self.SIZE[1]):
            self.done = True
            self.reward = self.conf["bonus_reward"]
        elif self.collision:
            self.done = True
            self.reward = self.conf["collision_reward"]
        elif self.out_of_bounds:
            self.done = True
            self.reward = self.conf["out_of_bounds_reward"]

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

        #render agents
        for agent_i in range(self.n_agents):
            #draw lidar fov
            if self.conf["viewer"]["draw_lidar"] == True:
                for neighbour in self.lidarsIndexes[agent_i]:# self.__get_neighbour_coordinates(self.agents[agent_i].pos ): #agent_pos[agent_i]):
                    if not self._full_obs[neighbour[0]][neighbour[1]] == PRE_IDS['wall']:
                        fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
                fill_cell(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            else:
                fill_cell(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='white')
            draw_cell_outline(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill='black',width=1)
            draw_circle(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=self.agents[agent_i].color)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agents[agent_i].pos, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)  
            
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
                printMap(self._full_obs, self.n_agents)
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
            # agent.collide = True
            # agent.silent = True
            #initialize other properties
            agent.color = AGENT_COLORS[i]
            
        return agents

    def reset_agents(self):
        for agent_i in range(self.n_agents):
            self.agents[agent_i].done = False
            self.agents[agent_i].pos = None
            self.agents[agent_i].collision = False
            self.agents[agent_i].out_of_bounds = False  
            #TODO ver o que mais é preciso dar reset??
    
    #CONFIRMAR! alterar para: um agent vê o seu explored map basicamente
    def get_agents_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            # pos = self.agents[agent_i].pos
            _agent_i_obs = self.agents[agent_i].exploredMap 
            # _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # # check if prey is in the view area
            # _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            # for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
            #     for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
            #         if PRE_IDS['prey'] in self._full_obs[row][col]:
            #             _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.

            # _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            # _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        # if self.full_observable:
        #     _obs = np.array(_obs).flatten().tolist()
        #     _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def get_agents_dones(self):
        return [self.agents[agent_i].done for agent_i in range(self.n_agents)]

    def _computeReward(self, agent_i):
        pastExploredCells = np.count_nonzero(self.agents[agent_i].pastExploredMap)
        currentExploredCells = np.count_nonzero(self.agents[agent_i].exploredMap)
        reward= 0

        #does this ever happen? since we check is the action is valid before it is taken? 
        if self.agents[agent_i].collision: 
            reward = self.conf["collision_reward"]
        if self.agents[agent_i].out_of_bounds: 
            reward = self.conf["out_of_bounds_reward"]

        #CONFIRM!!
        return reward + currentExploredCells - pastExploredCells - self._step_cost


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        # name 
        self.name = ''
        self.id = None
        self.pos = None
        self.done = False
        #each agent has its own explored map
        self.exploredMap = []
        self.pastExploredMap = []
        # agents are movable by default
        #self.movable = True
        # cannot send communication signals
        self.silent = False
        
        self.collision = False
        self.out_of_bounds = False
        # cannot observe the world
        #self.blind = False
        # physical motor noise amount
        #self.u_noise = None
        #communication range
        self.c_range = 3.0
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # color
        self.color = None
        # state
        self.state = None#AgentState() #simplificar? acho que pode ser útil para saber se está a comunicar ou non
        # action
        self.action = None #vou simplificar e vai ser um nº inteiro #Action()
        # script behavior to execute
        self.action_callback = None #-> TODO pôr a policy/model aqui??? 

    #it's considered in range, inside a square with distance of c_range squares around the agent
    def in_range(self, agent2):
        delta_pos = abs(np.subtract(np.array(self.pos), np.array(agent2.pos)) ) 

        if delta_pos[0] > self.c_range and delta_pos[1] > self.c_range :
            return False
        else:
            return True


AGENT_COLORS = [ImageColor.getcolor('blue', mode='RGB'),ImageColor.getcolor('red', mode='RGB'),ImageColor.getcolor('green', mode='RGB'),ImageColor.getcolor('yellow', mode='RGB')]
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
