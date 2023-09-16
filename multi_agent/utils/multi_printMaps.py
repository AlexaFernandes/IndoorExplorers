import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator, IndexLocator, MultipleLocator



# define color map 
color_map = {   0.0: np.array([211, 211, 211]), #lighgrey
                0.3: np.array([255,255,255]), # white
                0.5: np.array([0,0,0]),  
                1.0: np.array([0, 150, 255]),# blue 
                2.0: np.array([220, 10, 10]), #red
                3.0: np.array([0, 204, 0]), #green
                4.0: np.array([255, 215, 0]) #yellow
            }

colors1= ["#D7D7D7","#FFFFFF","#000000","#0096FF","#E11414" ,"#00CC00","#FFD700"]
labels1= ['unexplored','free','obstacle','UAV0','UAV1','UAV2','UAV3']

colors2 = ["#D7D7D7","#FFFFFF","#0096FF","#E11414" ,"#00CC00","#FFD700"]
labels2= ['unexplored','free','UAV0','UAV1','UAV2','UAV3']

#prints 1 square map
def printMap(matrix, n_agents):

    #define discrete colors for the colormap
    # cmap1 = ListedColormap(["lightgrey", "white", "black", "blue", "red", "green", "yellow"],"all_camp")
    # cmap2 = ListedColormap(["lightgrey", "white", "blue"],"no_obstacles_cmap")
    # cmap3 = ListedColormap([[211, 211, 211],[255,255,255],[0,0,0], [0, 150, 255],[225, 20, 20],[0, 204, 0],[255, 215, 0] ])

    cmap4= ListedColormap(colors1)
    cmap5= ListedColormap(colors2)
    
    lst = [0.0, 0.3, 0.6, 1.0, 2.0,3.0,4.0]
        
    fig, ax= plt.subplots()
    fig.set_tight_layout(True)
    #adding grid lines
    plt.hlines(y=np.arange(0, matrix.shape[1])+0.5, xmin=np.full(matrix.shape[1], 0)-0.5, xmax=np.full(matrix.shape[1], matrix.shape[1])-0.5, color="grey")
    plt.vlines(x=np.arange(0, matrix.shape[0])+0.5, ymin=np.full(matrix.shape[0], 0)-0.5, ymax=np.full(matrix.shape[0], matrix.shape[0])-0.5, color="grey")

    data_3d = np.ndarray(shape=(matrix.shape[0], matrix.shape[1], 3), dtype=int)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            data_3d[i][j] = color_map[matrix[i][j]]

    #print numbers inside squares
    # for i in range(matrix.shape[1]): #collumns
    #     for j in range(matrix.shape[0]): #rows
    #         c = matrix[j,i]
    #         ax.text(i, j, str("%.1f"%c), va='center', ha='center')

    ax.set_title("Updated map") #set title name
    values = np.unique(matrix.ravel())
    
    if 0.5 in matrix: #if an obstacle as been found the colormap should include the color blue and the colorbar should be accordingly
        pos=ax.imshow(data_3d,cmap=cmap4) #outra alternativa é o matshow, mas assim o titulo nao aparece

        #patches = [mpatches.Patch(color=colors1[i], label=labels1[i] ) for i in range(len(values)) ]
        patches = [mpatches.Patch(color=colors1[x], label=labels1[x] ) for x in range(n_agents+3) ]
        
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    else:
        pos=ax.imshow(data_3d,cmap=cmap5) #outra alternativa é o matshow, mas assim o titulo nao aparece

        patches = [ mpatches.Patch(color=colors1[i], label=labels2[i] ) for i in range(n_agents+2)  ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.show()


#TODO: INCOMPLETE! grid lines not right
def print2Map(matrix, n_agents, groundTruthMap):
    # colors1= ["#D7D7D7","#FFFFFF","#000000","#0096FF","#E11414" ,"#00CC00","#FFD700"]
    # labels1= ['unexplored','free','obstacle','UAV0','UAV1','UAV2','UAV3']

    # colors2 = ["#D7D7D7","#FFFFFF","#0096FF","#E11414" ,"#00CC00","#FFD700"]
    # labels2= ['unexplored','free','UAV0','UAV1','UAV2','UAV3']

    cmap4= ListedColormap(colors1)
    cmap5= ListedColormap(colors2)

    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)#,figsize=(12,2))
    fig.set_tight_layout(True)



    

    

    ax[0].set_title("Updated map") #set title name
    values = np.unique(matrix.ravel())

    data_3d = np.ndarray(shape=(matrix.shape[0], matrix.shape[1], 3), dtype=int)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            data_3d[i][j] = color_map[matrix[i][j]]

    if 0.5 in matrix: #if an obstacle as been found the colormap should include the color blue and the colorbar should be accordingly

        pos=ax[0].imshow(data_3d,cmap=cmap4) #outra alternativa é o matshow, mas assim o titulo nao aparece

        #patches = [mpatches.Patch(color=colors1[i], label=labels1[i] ) for i in range(len(values)) ]
        patches = [mpatches.Patch(color=colors1[x], label=labels1[x] ) for x in range(n_agents+3) ]
        
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        # cbar = plt.colorbar(pos) #make colorbar from the ploted imshow
        # cbar.ax.get_yaxis().set_ticks([])
        
        # for j, lab in enumerate(['$unexplored$','$free$','$obstacle$','$UAV0$','$UAV1$','$UAV2$','$UAV3$']):
        #     cbar.ax.text(1.0, (2 * j + 1) / 0.055, lab, ha='left', va='center')#0.055
    else:
        pos=ax[0].imshow(data_3d,cmap=cmap5) #outra alternativa é o matshow, mas assim o titulo nao aparece

        patches = [ mpatches.Patch(color=colors1[i], label=labels2[i] ) for i in range(n_agents+2)  ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        # cbar = plt.colorbar(pos) #make colorbar from the ploted imshow
        # cbar.ax.get_yaxis().set_ticks([])
        # for j, lab in enumerate(['$unexplored$','$free$','$UAV0$','$UAV1$','$UAV2$','$UAV3$']):
        #     cbar.ax.text(0.8, (2 * j + 1) / 10.0, lab, ha='left', va='center')

    #adding grid lines
    plt.hlines(y=np.arange(0, groundTruthMap.shape[1])+0.5, xmin=np.full(groundTruthMap.shape[1], 0)-0.5, xmax=np.full(groundTruthMap.shape[1], groundTruthMap.shape[1])-0.5, color="grey")
    plt.vlines(x=np.arange(0, groundTruthMap.shape[0])+0.5, ymin=np.full(groundTruthMap.shape[0], 0)-0.5, ymax=np.full(groundTruthMap.shape[0], groundTruthMap.shape[0])-0.5, color="grey")

    ax[1].set_title("Ground Truth Map") #set title name
    #values = np.unique(groundTruthMap.ravel())

    data_3d = np.ndarray(shape=(groundTruthMap.shape[0], groundTruthMap.shape[1], 3), dtype=int)
    for i in range(0, groundTruthMap.shape[0]):
        for j in range(0, groundTruthMap.shape[1]):
            data_3d[i][j] = color_map[groundTruthMap[i][j]]

    pos=ax[1].imshow(data_3d,cmap=cmap5) #outra alternativa é o matshow, mas assim o titulo nao aparece

    # patches = [ mpatches.Patch(color=colors1[i], label=labels2[i] ) for i in range(n_agents+2)  ]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


    # for i, ax in enumerate(axn.flat):
    #     k = list(cf_matrix)[i]
    #     sns.heatmap(cf_matrix[k], ax=ax,cbar=i==4)
    #     ax.set_title(k,fontsize=8)

    plt.show()



def printAgentsMaps(agents, n_agents):
    # colors1= ["#D7D7D7","#FFFFFF","#000000","#0096FF","#E11414" ,"#00CC00","#FFD700"]
    # labels1= ['unexplored','free','obstacle','UAV0','UAV1','UAV2','UAV3']

    # colors2 = ["#D7D7D7","#FFFFFF","#0096FF","#E11414" ,"#00CC00","#FFD700"]
    # labels2= ['unexplored','free','UAV0','UAV1','UAV2','UAV3']

    cmap4= ListedColormap(colors1)
    cmap5= ListedColormap(colors2)

    fig, ax = plt.subplots(1,n_agents, figsize=(5 * n_agents, 4.5), sharex=True, sharey=True)#,figsize=(12,2))
    fig.set_tight_layout(True)

    grid_shape = agents[0].exploredMap.shape


    for agent_i in range(n_agents):
        ax[agent_i].set_title("Agent {} map".format(agent_i)) #set title name
        matrix = agents[agent_i].exploredMap

        values = np.unique(matrix.ravel())
        data_3d = np.ndarray(shape=(grid_shape[0], grid_shape[1], 3), dtype=int)

        for i in range(0, grid_shape[0]):
            for j in range(0, grid_shape[1]):
                data_3d[i][j] = color_map[matrix[i][j]]

        if 0.5 in matrix: #if an obstacle as been found the colormap should include the color blue and the colorbar should be accordingly
            pos=ax[agent_i].imshow(data_3d,cmap=cmap4) #outra alternativa é o matshow, mas assim o titulo nao aparece

        else:
            pos=ax[agent_i].imshow(data_3d,cmap=cmap5) #outra alternativa é o matshow, mas assim o titulo nao aparece

        # for i in range(matrix.shape[1]): #collumns
        #     for j in range(matrix.shape[0]): #rows
        #         c = matrix[j,i]
        #         ax[agent_i].text(i, j, str("%.1f"%c), va='center', ha='center')

        #adding grid lines
        # Set minor ticks/gridline cadence
        ax[agent_i].yaxis.set_minor_locator(IndexLocator(base=1.0, offset=0.0))
        ax[agent_i].xaxis.set_minor_locator(IndexLocator(base=1.0, offset=0.0))
        ax[agent_i].grid(which = "minor", linewidth=1.4)
        #change tick color to white
        ax[agent_i].tick_params(which = "minor" ,axis='both', colors='white')

        ax[agent_i].yaxis.set_major_locator(IndexLocator(base=1.0, offset=0.5))
        ax[agent_i].xaxis.set_major_locator(IndexLocator(base=1.0, offset=0.5))

    patches = [mpatches.Patch(color=colors1[x], label=labels1[x] ) for x in range(n_agents+3) ]
    
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    
    plt.show()

