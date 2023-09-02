import copy
import logging

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

import multi_agent.core
from multi_agent.utils.printMaps import printMap
from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.utils.multi_agent_lidarSensor import Lidar

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

logger = logging.getLogger(__name__)

RANDOM_SPAWN=False #TODO colocar no ficheiro depois

class IndoorExplorers(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, conf=None, grid_shape=(21, 21),# n_agents=4,
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5 ): #, max_steps=100,
                 #agent_view_mask=(5, 5)):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        # assert len(agent_view_mask) == 2, 'expected a tuple of size 2 for agent view mask,' \
        #                                   ' but found {}'.format(agent_view_mask)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        # assert 0 < agent_view_mask[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        # assert 0 < agent_view_mask[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])

        self._grid_shape = grid_shape #tem os mesmos valores que conf["size"]
        self.n_agents = conf["n_agents"] 
        self.agents = self.create_agents() #TODO verificar se é preciso inicializar mais alguma coisa nesta função
        self._max_steps = conf["max_steps"]
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._agent_view_mask = agent_view_mask

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)]) #por agora vou deixar 5 ações (L R U D NOOP) TODO: add share info
        #self.agent_pos = {_: None for _ in range(self.n_agents)} 
        #self.exploredMaps = [None for _ in range(self.n_agents)] #explored map of every agent
        #self._agent_dones = [False for _ in range(self.n_agents)] #this is done when creating the agents

        self.lidar_map = None # with no agents, just the walls -> 0.0 -> unexplored , 0.5 walls
        self.groundTruthMap = None #map fully explored aka the ground truth to be compared  to -> 0.3 explored , 0.5 walls
        self._full_obs = self.__create_grid() #map wit overall view from every agent - the global map
        
        self.viewer = None
        self.full_observable = full_observable

        #observation space is the view of the map
        # 0.0 --> unexplored
        # 0.3 --> explored/empty
        # 0.5 --> obstacle
        # >1.0 -> agents ids
        self._obs_high = np.array(self.n_agents, dtype=np.float32)
        self._obs_low = np.array(0.0, dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high, grid_shape) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()


    #checks if is a wall in pos
    def __wall_exists(self, pos):
        row, col = pos
        return 1.0 in self.groundTruthMap[row,col]
        #return PRE_IDS['wall'] in self._base_grid[row, col]

    #checks if cell is valid (aka inside bounds)
    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    #checks if cell is vacant
    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    #check is there is any other agents nearby
    #def look_for_other_agents(self, pos): #pos or better agent?
        #TODO
        #returns list or something with id and pos of each agent

    def is_collision(self, agent1, agent2):
        #TODO reescrever
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    #função que atualiza a info de cada agente 
    def __update_agent_view(self, agent_i):

        # initialing position is explored
        self._activateLidars()

        #update its own explored map
        self._updateMaps()

        #update de ground_truth_map 
        self._full_obs[self.agents[agent_i].pos[0]][self.agents[agent_i].pos[1]] = PRE_IDS['agent'] + str(agent_i + 1)
        #self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)
        #dont worry that the the previous pos of the agent is set to 0 inside __update_aget_pos()

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

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agents[agent_i].pos = next_pos #agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

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

    #TODO alterar para: um agent vê o seu explored map basicamente
    def get_agents_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agents[agent_i].pos
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if prey is in the view area
            _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.

            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def __create_grid(self):
        # _full_obs --> 0.5 obstacle
        #               0.3 free to move
        #               0.0 unexplored
        #               >1  agent id
        gen = Generator(self.conf)
        randomMap = gen.get_map().astype(np.double)

        #save lidar map
        self.lidar_map = randomMap.copy() #save a map not explored, with 0.0 (unexplored) and 1.0 (walls)

        randomMap[randomMap == 1.0] = 0.5 #obstacle value is 0.5
        randomMap[randomMap == 0.0] = 0.3 #explored cells value is 0.3
        self.groundTruthMap = randomMap.copy() #save the ground truth, fully explored
        return randomMap

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=UNEXPLORED_COLOR)

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
            self.pastExploredMaps[agent_i] = self.agents[agent_i].exploredMap.copy() #exploredMaps[agent_i].copy()

            lidarX= self.lidarsIndexes[agent_i][:,0]
            lidarY = self.lidarsIndexes[agent_i][:,1]
            self.agents[agent_i].exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY] #exploredMaps[agent_i][lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]

            self.agents[agent_i].exploredMap[self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]] = agent_i 
            #self.agents[agent_i].exploredMap[self.agent_pos[agent_i][0],[self.agent_pos[agent_i][1]]] = agent_i 

    #CONFIRM!!
    def _activateLidars(self):

        for agent_i in range(self.n_agents): 
            self.ldr[agent_i].update(self.agents[agent_i].pos) #TODO CONFIRMAR!! alternativa: [self.agents[agent_i].pos[0],[self.agents[agent_i].pos[1]]]
            thetas[agent_i], ranges[agent_i] = self.ldr[agent_i].thetas, self.ldr[agent_i].ranges
            indexes[agent_i] = self.ldr[agent_i].idx

        self.lidarsIndexes = indexes


    #CONFIRMAR!! esta vai ser a função de inicialização do mapa e das posições de cada agente
    def __init_full_obs(self):
        #creates new map
        self._full_obs = self.__create_grid()

        self.create_lidars()

        # create an empty exploredMap for each agent and
        # inserts agents at random locations #TODO mudar para que dê spawn nos cantos do mapa
        for agent_i in range(self.n_agents):
            # 0 if not visible/visited, 1 if visible/visited
            self.agents[agent_i].exploredMap = np.zeros(self.SIZE, dtype=np.double) #exploredMaps[agent_i] = np.zeros(self.SIZE, dtype=np.double)

            if RANDOM_SPAWN: random_spawn=True
            while True:
                if random_spawn:
                    pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                        self.np_random.randint(0, self._grid_shape[1] - 1)]
                else:
                    pos = conf["initial"][agent_i]

                if self._is_cell_vacant(pos):
                    self.agents[agent_i].pos = pos #agent_pos[agent_i] = pos
                    break
                else:
                    #set flag to generate a new possible spawn position
                    random_spawn=True
            #reset spawn flag
            random_spawn=RANDOM_SPAWN
            
            #TODO será melhor dar upadte individual??
            #self.__update_agent_view(agent_i) 
        
        #OUU fazer assim:
        # initial position is explored
        self._activateLidars()

        #update everyone's explored maps
        self._updateMaps()

        #create grid for later render
        self.__draw_base_img()


    #TODO
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

        return self.get_agent_obs()

    #CONFIRMAR!!
    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        #TODO compute reward 
        rewards = [self._step_cost for _ in range(self.n_agents)]

        #apply chosen action
        for agent_i, action in enumerate(agents_action):
            if not (self.agents[agent_i].done): #_agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
        
        #check if max steps as been reached or the all map as been explored
        if (self._step_count >= self._max_steps) or (0.0 not in self._full_obs):  #old: or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self.agents[i].done = True  #_agent_dones[i] = True

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

        # printMap(_full_obs)

        return self.get_agent_obs(), rewards, self.get_agents_dones()  #_agent_dones
        #the info parameter was: {'prey_alive': self._prey_alive} in the original code, TODO see what extra indo would be useful for me

    #TODO
    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        printMap(_full_obs)

        img = copy.copy(self._base_img)

        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agents[agent_i].pos ): #agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for col in range(0, _grid_shape[1]):
            for row in range(0, _grid_shape[0]):
                pos = [row,col]
                if _full_obs[row][col] ==  0.3: #explored cells
                    fill_cell(img,pos, cell_size=CELL_SIZE, fill='white')
                if _full_obs[row][col] == 0.5: #walls
                    fill_cell(img,pos, cell_size=CELL_SIZE, fill=WALL_COLOR)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agents[agent_i].pos, cell_size=CELL_SIZE, fill=agents[agent_i].color)
            #old: draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])
            write_cell_text(img, text=str(agent_i + 1), pos=self.agents[agent_i].pos, cell_size=CELL_SIZE,
                            fill='white', margin=0.4)  
            #old: write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.4) 

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
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
            self.agents.done = False
            self.agents.pos = None
            #TODO ver o que mais é preciso dar reset??
    
    def get_agents_dones():
        return [agent.done for agent in agents]


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
        # cannot observe the world
        #self.blind = False
        # physical motor noise amount
        #self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # color
        self.color = None
        # state
        self.state = AgentState() #simplificar? acho que pode ser útil para saber se está a comunicar ou non
        # action
        self.action = None #vou simplificar e vai ser um nº inteiro #Action()
        # script behavior to execute
        self.action_callback = None #-> TODO pôr a policy/model aqui??? 

        


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
    'wall': 'W',
    'unexplored': 0.0,
    'empty': 0.3
}
