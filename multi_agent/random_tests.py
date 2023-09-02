# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Online Python-3 Compiler (Interpreter)
from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.settings import DEFAULT_CONFIG as conf
import numpy as np
from multi_agent.utils.multi_printMaps import printMap

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        # name 
        self.name = ''
        self.id = None
        #each agent has its own explored map
        self.exploredMap = []
        self.pastExploredMap = []
        self.done = False
        self.pos = [0,0]
        # color
        self.color = None
        # state
        # action
        self.action = None #vou simplificar e vai ser um nº inteiro #Action()
        # script behavior to execute
        self.action_callback = None #-> TODO pôr a policy/model aqui??? 

def change_dones(agents):
    for i, agent in enumerate(agents):
        if i%2==0:
            agent.done=True

    return [agent.done for agent in agents] 

def get_agents_dones():
    return [agent.done for agent in agents]

def __wall_exists(self, pos):
    row, col = pos
    return 1.0 in self.groundTruthMap
    


if __name__ == "__main__":
    agents = [Agent() for i in range(3)]
    for i, agent in enumerate(agents):
        agent.name = 'agent %d' % i
        agent.id = i
        agent.pos=[i,i]
        
        # agent.collide = True
        # agent.silent = True
        #initialize other properties
    
    gen = Generator(conf)
    groundTruthMap = gen.get_map().astype(np.double)
    # [[1.0, 0.3 ,0.3] ,
    #                   [0.3, 1.0, 0.3],
    #                   [1.0, 0.3, 0.3]
    #                   ]       
    
    # for agent_i in range (3):
    #     print("{} is in {}".format(agents[agent_i].id, agents[agent_i].pos))
    #     agents[agent_i].pos = [0,0]
    #     print("{} is in {}".format(agents[agent_i].id, agents[agent_i].pos))

    #print(groundTruthMap)
    #print("{}".format([1.0 in groundTruthMap]))

    # _full_obs =np.array([ [1.0, 0.3, 0.3 , 2.0],
    #                       [0.3, 0.3, 0.3, 0.3 ],
    #                       [0.5, 0.5, 0.5, 0.5],
    #                       [3.0, 0.0, 0.0, 4.0]
    #             ])

    _full_obs =np.array([ [1.0, 0.3, 0.3 , 2.0],
                          [0.3, 0.3, 0.3, 0.3],
                          [0.5, 0.5, 0.5, 0.5],
                          [3.0, 0.0, 0.0, 4.0]
                ])
    _grid_shape = (3,4)
    print(_full_obs.shape)

    printMap(_full_obs)

    # for col in range(0, _grid_shape[1]):
    #     for row in range(0, _grid_shape[0]):
    #         if _full_obs[row][col] == 0.0:
    #             print("explored", row, col)
    #         elif _full_obs[row][col] == 0.3:
    #             print("unexplored",row, col)
    #         elif _full_obs[row][col] == 0.5:
    #             print("wall", row, col)
    #         else:
    #             print("A"+str(_full_obs[row][col]))
    #print("{}".format(get_agents_dones()))
        
    