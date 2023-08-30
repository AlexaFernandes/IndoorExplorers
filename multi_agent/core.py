import numpy as np

from indoor_explorers.utils.printMaps import printMap
from indoor_explorers.utils.randomMapGenerator import Generator
from indoor_explorers.utils.lidarSensor import Lidar
from indoor_explorers.render.viewer import Viewer
from indoor_explorers.envs.settings import DEFAULT_CONFIG

# physical/external base state of all entites
# class EntityState(object):
#     def __init__(self):
#         # physical position
#         self.p_pos = None
#         # physical velocity
#         self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(object):
    def __init__(self):
        super(AgentState, self).__init__()
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None #TODO do I need this?
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
# class Entity(object):
#     def __init__(self):
#         # name 
#         self.name = ''
#         # properties:
#         self.size = 0.050
#         # entity can move / be pushed
#         self.movable = False
#         # entity collides with others
#         self.collide = True
#         # material density (affects mass)
#         self.density = 25.0
#         # color
#         self.color = None
#         # max speed and accel
#         self.max_speed = None
#         self.accel = None
#         # state
#         self.state = EntityState()
#         # mass
#         self.initial_mass = 1.0

#     @property
#     def mass(self):
#         return self.initial_mass


# properties of agent entities
class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        # name 
        self.name = ''
        self.id = None
        #each agent has its own explored map
        self.explored_map = []
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # color
        self.color = None
        # state
        self.state = AgentState()
        # action
        self.action = None #vou simplificar #Action()
        # script behavior to execute
        self.action_callback = None #-> TODO pôr a policy/model aqui??? 

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and ground truth map 
        self.agents = []
        self.map_dim = []
        self.ground_truth_map = []
        self.lidar_map = []
        #list of obstacles
        self.obstacles_idx = []
        # communication channel dimensionality
        self.com_range = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        #self.dt = 0.1

    # return all agents controllable by external policies
    @property
    def get_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.get_agents:
            agent.action = agent.action_callback(agent, self)
        
        #update map for each agent
        #e mais cenas
        
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    def make_world(grid_shape, num_agents): 
        # set any world properties first
        world.com_range = 5
        world.map_dim = grid_shape
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            # agent.collide = True
            # agent.silent = True
            
        # create map
        

        #save location of obstacles

        # make initial conditions
        self.reset_world()

    def reset_world():
        #clean all buffers
        #clean all matrixes
        #create new map
        #update all info for the new one
        return 
