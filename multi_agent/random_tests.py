# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Online Python-3 Compiler (Interpreter)
from multi_agent.utils.randomMapGenerator import Generator
from multi_agent.utils.multi_agent_lidarSensor import Lidar
from multi_agent.settings import DEFAULT_CONFIG as conf
import numpy as np
import random
import itertools
from multi_agent.utils.multi_printMaps import printMap
from itertools import combinations


PRE_IDS = {
    'agent': 'A',
    'wall': 0.5,
    'unexplored': 0.0,
    'empty/explored': 0.3
}


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
        self.collision = False
        self.out_of_bounds = False
        self.c_range = 1.0
        # color
        self.color = None
        # state
        # action
        self.action = None #vou simplificar e vai ser um nº inteiro #Action()
        # script behavior to execute
        self.action_callback = None #-> TODO pôr a policy/model aqui??? 

    def is_alive(self):
        return (not self.collision) and (not self.out_of_bounds) 

    def does_wall_exists( self,pos):
        row, col = pos
        return (1.0 in self.exploredMap[row,col])

    def in_range(self, agent2):
        delta_pos = abs(np.subtract(np.array(self.pos), np.array(agent2.pos)) ) 
        #delta_pos = self.pos - agent2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        #print(delta_pos)

        if delta_pos[0] > self.c_range or delta_pos[1] > self.c_range :
            return False
        else:
            return True

def change_dones(agents):
    for i, agent in enumerate(agents):
        if i%2==0:
            agent.done=True

    return [agent.done for agent in agents] 

def get_agents_dones():
    return [agent.done for agent in agents]

def is_valid(mat, pos):
        return (0 <= pos[0] < mat.shape[0]) and (0 <= pos[1] < mat.shape[1])

def _is_cell_vacant(mat , pos):
        return (is_valid(mat,pos) and (mat[pos[0]][pos[1]] == PRE_IDS['empty/explored']))

def is_alive(agents, agent_i):
        return (not agents[agent_i].collision) and (not agents[agent_i].out_of_bounds)

def merge_maps(maps, agent_list):
        new_merged_map = np.full(maps[0].shape, 0.0)

        for col in range(0, maps[0].shape[1]):
            for row in range(0, maps[0].shape[0]):
                for agent_i in agent_list:
                    if maps[agent_i][row][col] != 0.0:
                        new_merged_map[row][col]= maps[agent_i][row][col]

        # for agent_i in agent_list:
        #     self.agents[agent_i].exploredMap = new_merged_map.copy()
        return new_merged_map

def update_comm_range(agents, n_agents):
    l = []
    l.extend(range(0, n_agents))
    #generate a list with all the unique pairings among all agents
    combinations = list(itertools.combinations(l, 2))
    in_range = []
    comm_range = np.full((n_agents,n_agents), False)

    #print(combinations)
    #check for collisions between any 2 agents
    for pair in combinations:
        if agents[pair[0]].in_range(agents[pair[1]]) == True:
            in_range.append(pair)
            comm_range[pair[0]][pair[1]] = comm_range[pair[1]][pair[0]] = 1 
        else:
            comm_range[pair[0]][pair[1]] = comm_range[pair[1]][pair[0]] = 0

    #return list of agents in range
    return comm_range #,in_range 


    def DFSUtil(matrix, temp, agent_i, visited, n_agents):
 
        # Mark the current vertex as visited
        visited[agent_i] = True

        # Store the vertex to list
        temp.append(agent_i)

        # Repeat for all vertices adjacent
        # to this vertex v
        for j in range(0, n_agents):
            #go through the upper triangular matrix (since the matrix is simmetric, we olny need to go through one of the triangular matrices)
            if (agent_i < j):
                if matrix[agent_i][j] == 1: #if they are connected
                    if visited[j] == False:
                        # Update the list
                        temp = DFSUtil(matrix,temp, j, visited, n_agents)
                    
        return temp


def DFSUtil(matrix, temp, v, visited, n_agents):
 
    # Mark the current vertex as visited
    visited[v] = True

    # Store the vertex to list
    temp.append(v)

    # Repeat for all vertices adjacent
    # to this vertex v
    for j in range(0,n_agents):
        #go through the upper triangular matrix (since the matrix is simmetric, we olny need to go through one of the triangular matrices)
        if (v < j):
            if matrix[v][j] == 1: #if they are connected
                if visited[j] == False:
                    # Update the list
                    temp = DFSUtil(matrix,temp, j, visited, n_agents)
                
    return temp


def connectedComponents(matrix, n_agents):
    visited = []
    cc = []
    for i in range(n_agents):
        visited.append(False)
    for agent_i in range(n_agents):
        if visited[agent_i] == False:
            temp = []
            cc.append(DFSUtil(matrix, temp, agent_i, visited, n_agents))
    return cc

if __name__ == "__main__":
    n_agents=4
    agents = [Agent() for i in range(n_agents)]
    for i, agent in enumerate(agents):
        agent.name = 'agent %d' % i
        agent.id = i
        agent.pos=[i,i]
        
        # agent.collide = True
        # agent.silent = True
        #initialize other properties
    
    # agents[1].pos=[4,0]
    # agents[2].pos=[4,1]
    # agents[3].pos=[4,2] 
    #agents[0].in_range

    gen = Generator(conf)
    #groundTruthMap = gen.get_map().astype(np.double)
    groundTruthMap=  np.array([[0.3, 0.3 ,0.3, 0.5, 0.3, 0.3] ,
                               [0.3, 0.3, 0.3, 0.5, 0.3, 0.3],
                               [0.3, 0.3, 0.3, 0.5, 0.3, 0.3],
                               [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                               [0.3, 0.3, 0.3, 0.5, 0.3, 0.3],
                               [0.3, 0.3, 0.3, 0.5, 0.3, 0.3]
                                ]  )  
    # #printMap(groundTruthMap,n_agents)
    _grid_shape = groundTruthMap.shape
    lidar_map = groundTruthMap.copy()
    lidar_map[lidar_map == 0.5] = 1.0 #obstacle value is 0.5
    lidar_map[lidar_map == 0.3] = 0.0 #explored cells, which value is 0.3
    #print(np.matrix(lidar_map))
    #print(conf["lidar_range"])
    #CREATE LIDAR
    #create an array of lidar, one per agent
    ldr = [Lidar(r=conf["lidar_range"],
                        channels=conf["lidar_channels"],
                        _map=lidar_map)                   for _ in range(n_agents)]
    #print(ldr[0].maxRange)
    #create list of obstacle indexes
    # obstacles_idx = np.where(groundTruthMap == 0.5)
    # obstacles_x = obstacles_idx[0]
    # obstacles_y = obstacles_idx[1]
    # obstacles_idx = np.stack((obstacles_x, obstacles_y), axis=1)
    # obstacles_idx = [list(i) for i in obstacles_idx]
    

    #ACTIVATE LIDAR 
    indexes = [None for _ in range(n_agents)]
    for agent_i in range(n_agents): 
        #print("LIDAR: is alive:{} ,agent {}, with pos{}".format(self.agents[agent_i].is_alive(), agent_i, self.agents[agent_i].pos))
        ldr[agent_i].update(agents[agent_i].pos)
        #thetas[agent_i], ranges[agent_i] = self.ldr[agent_i].thetas, self.ldr[agent_i].ranges
        indexes[agent_i] = ldr[agent_i].idx
    
    lidarX = indexes[0][:,0]
    lidarY = indexes[0][:,1]
    #lidar_idx = np.stack((lidarX, lidarY), axis=1)

    row=1
    col=0
    agent_i=0
    for pos in indexes[0]:
        if list(pos) == [row,col]:
            print("{} can see{}".format(agent_i, pos))
            break
        # if lidar_map[pos[0]][pos[1]] == 1.0:
        #     print("[{}] Wall".format(pos))
        # elif list(pos) == [row,col]:
        #     print("{}  HERE".format(pos))
        # elif list(pos) == agents[0].pos:
        #     print("{} - agent 0 pos".format(pos))
        # elif list(pos) == agents[1].pos:
        #     print("{} - agent 1 pos".format(pos))
        # else:
        #     print("[{}] FREE".format(pos))
    # for col in range(0, _grid_shape[1]):
    #         for row in range(0, _grid_shape[0]):
    #             if [row,col] not in indexes[0]:
    #                 print("[{},{}] not in range".format(row,col))
    
    # printMap(groundTruthMap,n_agents)
    # [[1.0, 0.3 ,0.3] ,
    #                   [0.3, 1.0, 0.3],
    #                   [1.0, 0.3, 0.3]
    #                   ]       
    
    # for agent_i in range (4):
    #     print("{} is in {}".format(agents[agent_i].id, agents[agent_i].pos))
        # agents[agent_i].pos = [0,0]
        # print("{} is in {}".format(agents[agent_i].id, agents[agent_i].pos))

    # print(groundTruthMap)
    # print("{}".format([1.0 in groundTruthMap]))
    # agents[0].exploredMap = groundTruthMap.copy()

    #value = agents[0].does_wall_exists()
    #print(value)

   
    # print((_full_obs==groundTruthMap).all())

    # _full_obs =np.array([ [1.0, 0.3, 0.3 , 2.0],
    #                       [0.3, 0.3, 0.3, 0.3],
    #                       [0.5, 0.5, 0.5, 0.5],
    #                       [3.0, 0.0, 0.0, 0.0]
    #             ])
    # _grid_shape = (3,4)
    # print(_full_obs.shape)

    # printMap(_full_obs,3)




    #-----------------------------------------------------------
    # mat1 =np.array([[1.0, 0.3, 0.0 , 0.0],
    #                 [0.3, 0.3, 0.0, 0.0 ],
    #                 [0.0, 0.0, 0.0, 2.0],
    #                 [0.0, 0.0, 0.0, 0.0]
    #             ])

    #print(_is_cell_vacant(mat1, [2,2]))

    #free_x, free_y = np.where(mat1 == 0.3)
    #print(free_x,free_y)#, free_z)

    # mat2 =np.array([ [0.0, 0.0, 0.3 , 2.0],
    #                  [0.0, 0.0, 4.0, 0.3 ],
    #                  [0.0, 0.0, 0.0, 0.0],
    #                  [0.0, 0.0, 0.0, 0.0]
    #             ])

    # mat3 =np.array([ [0.0, 0.0, 0.0 , 0.0],
    #                  [0.0, 0.0, 0.0, 0.0 ],
    #                  [0.5, 0.5, 0.0, 0.0],
    #                  [3.0, 0.3, 0.0, 0.0]
    #             ])

    # maps= [mat1,mat2,mat3]

    # observation_n = np.expand_dims(maps, axis=0) 
    # print(observation_n.shape)
    

    # agent_list = [0,1,2]

    # merged_map = merge_maps(maps, agent_list)
    # print(merged_map)

    # mat1 = merged_map
    # print(mat1)
    # mat1[1,2] = 0.0
    # mat4 = merged_map.copy()
    # print(mat4)
    # mat4[1,2] = 4.0

    # print(mat4)

    # print(merged_map)
    #-----------------------------------------------------------
    
    

    



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



    #------------------------------------------------
    #test connectivity:
    # matrix =  [ [0, 1, 0, 0],
    #             [1, 0, 1, 0],
    #             [0, 1, 0, 1],
    #             [0, 0, 1, 0]
    #             ]
    
    # matrix2 =  [[0, 0, 0, 0],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 1],
    #             [0, 0, 1, 0]
    #             ]

    # matrix3=  np.array([ [0, 1, 1, 1],
    #                      [1, 0, 0, 0],
    #                      [1, 0, 0, 0],
    #                      [1, 0, 0, 0]
    #                      ])
    # mat=matrix2         
    # print(mat)
    # comm_range = update_comm_range(agents,4)
    
    # print("Groups:")
    # groups=[]
    # groups = connectedComponents(mat,n_agents=4)
    # for group in groups:
    #     print(len(group))
    #     print(group)


#-----------------------------------------------------------------------
    #SHUFFLE order for action taking

    # action_n=[4,5,6,7]
    # idxs = np.arange(0,len(action_n))
    # action_dict = dict(enumerate(action_n))
    # print(action_dict)

    # random.shuffle(idxs)
    # print(idxs)

    # for i in range(len(action_n)):
    #     print(action_dict[idxs[i]])


#-----------------------------------------------------------------
import operator as op
import pandas as pd
from collections import Counter
 
# declaring the list
l = [[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1],[1,1], [1,1], [1,2], [2,0], [2,2], [3,0]]
l2= [(4,1),(5,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,2),(1,2),(1,2),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,2)]
x=[1,1]
#--------------------------------------
#to count specific
#print(f"{x} has occurred {op.countOf(l, x)} times")
#-------------------------------------------
#cool but overkill do a dataframe, not easy to ad of remove elements
df= pd.DataFrame(l)
count2= df.value_counts().index.tolist()
most_freq_elem=count2[0]

# print(most_freq_elem)
# print(df.value_counts()[most_freq_elem] )
#---------------------------
#cool but overkill do a series, not easy to ad of remove elements
# ser=pd.Series(l)
# count = ser.value_counts().index.tolist()

# most_freq_elem=count[0]
#print(most_freq_elem)
#print(ser.value_counts()[most_freq_elem] )
#-------------------------------
#THIS METHOD IS THE ONE THAT WORKS BETTER FOR WHAT I WANT
# import statistics
# from statistics import mode
# print(l2)
# print(mode(l2))
# print(op.countOf(l2, mode(l2)))
# pos=[3,0]
# l2.pop(0)
# l2.append(tuple([6,0]))
# l2.pop(0)
# l2.append(tuple(pos))
# l2.pop(0)
# print(l2)
# print(mode(l2))
# print(op.countOf(l2, mode(l2)))


#--------------------------------------------------------------------------

# agents[0].collision=False
# agents[0].out_of_bounds=True

# print(agents[0].is_alive())

#--------------------------------------------------------------------------
#test how to get coordinates of agent if it exists
# mat1 =np.array([ [0.0, 0.0, 0.0 , 0.0],
#                      [0.0, 0.0, 1.0, 0.3 ],
#                      [0.0, 0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0, 0.0]
#                 ])

# mat2 =np.array([ [0.0, 0.0, 0.3 , 2.0],
#                      [0.0, 0.0, 4.0, 0.3 ],
#                      [0.0, 0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0, 0.0]
#                 ])

# mat=mat1
# agent = 1
# agent_i= agent-1
# x,y = np.where(mat==agent_i+1)
# coord = np.array(list(zip(x, y)))
# agent_pos=[1,2]

# l = []
# l.extend(range(0, n_agents))
# #generate a list with all the unique pairings among all agents
# combinations = list(itertools.combinations(l, 2))
# print(combinations)

# print(coord)
# if len(coord) == 0:
#     print("empty")
# elif len(coord) == 1:
#     #check if corresponding pos is right
#     pos=coord[0]
#     if list(pos) == agent_pos: #agents[agent_idx].pos:
#         print("alright")
#     else:
#         print("u have the wrong pos, fix it!")
#         mat[pos[0]][pos[1]]=0.3 #mark as explored
#         mat[agent_pos[0]][agent_pos[1]]= agent_i + 1

# else:
#     #há mais do que 1 ocorrencia
#     #comparar com a pos atual e pôr as outras a "empty"
#     for pos in coord:
#         if list(pos) == agent_pos: #agents[agent_idx].pos:
#             print("this is the right pos {}".format(pos))
#         else:
#             mat[pos[0]][pos[1]]=0.3 #mark as explored

# print(mat)

#---------------------------------------
#testing continue statement
# for agent_i in range(n_agents):
#         print(agent_i)

#         #verify if for each agent pos in agent_i's map there is only 1 entrance of every other agent (agent_j)
#         #and if that entrance is correct
#         for agent_j in range(n_agents): 
#             # if agent_i != agent_j:
#             #     print("{}-{}".format(agent_i,agent_j))
#             # else:
#             #     print("equal")
#             if agent_i == agent_j:
#                 continue
#                 print("equal")
#             else:
#                 print("{}-{}".format(agent_i,agent_j))