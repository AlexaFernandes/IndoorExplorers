import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#prints 1 square map
def printMap(matrix):

    #define discrete colors for the colormap
    cmap1 = ListedColormap(["lightgrey", "white", "red", "darkblue"],"all_camp")
    cmap2 = ListedColormap(["lightgrey", "white", "red"],"no_obstacles_cmap")
    lst = [0.0, 0.3, 0.6, 1.0]
    flag=False

    # minima = min(lst)
    # maxima = max(lst)

    # norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    # mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # for v in sorted(lst):
    #     print("%.4f: %.4f" % (v, mapper.to_rgba(v)[0]) )
        

    fig, ax= plt.subplots()
    fig.tight_layout()

    #print numbers inside squares
    for i in range(matrix.shape[1]): #collumns
        for j in range(matrix.shape[0]): #rows
            c = matrix[j,i]
            if c == 1.0:
                flag=True
            ax.text(i, j, str("%.1f"%c), va='center', ha='center')

    ax.set_title("Ground Truth Map") #set title name
    
    if flag:
        pos=ax.imshow(matrix,cmap=cmap1) #outra alternativa é o matshow, mas assim o titulo nao aparece

        cbar = plt.colorbar(pos) #make colorbar from the ploted imshow
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$unexplored$','$free$','$UAV$','$obstacle$']):
            cbar.ax.text(1.25, (2 * j + 1) / 8.0, lab, ha='left', va='center')
    else:
        pos=ax.imshow(matrix,cmap=cmap2) #outra alternativa é o matshow, mas assim o titulo nao aparece

        cbar = plt.colorbar(pos) #make colorbar from the ploted imshow
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['$unexplored$','$free$','$UAV$']):
            cbar.ax.text(0.8, (2 * j + 1) / 10.0, lab, ha='left', va='center')
    
    plt.show()