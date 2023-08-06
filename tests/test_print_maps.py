import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import seaborn as sns


rows, cols = 6, 7

#intersection_matrix=np.zeros(21,21)
labels=["unexplored", "free", "UAV", "obstacles"]
intersection_matrix = np.array([[0.3, 0.3, 0.6, 0.3, 0.3, 0.0, 0.0],
                                [0.2, 0.3, 0.3, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.3, 0.3, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])#,
                                #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

#define discrete colors for the colormap
cmap = ListedColormap(["lightgrey", "white", "red", "darkblue"])


# #window 1
fig, ax= plt.subplots()
fig.tight_layout()

#print numbers inside squares
for i in range(intersection_matrix.shape[1]):
    for j in range(intersection_matrix.shape[0]):
        c = intersection_matrix[j,i]
        ax.text(i, j, str("%.1f"%c), va='center', ha='center')

ax.set_title("Ground Truth Map") #set title name
pos=ax.imshow(intersection_matrix,cmap=cmap) #outra alternativa é o matshow, mas assim o titulo nao aparece


cbar = plt.colorbar(pos) #make colorbar from the ploted imshow
cbar.ax.get_yaxis().set_ticks([])
for j in labels.len:
    cbar.ax.text(1.25, (2 * j + 1) / 8.0, labels[j], ha='left', va='center')



plt.show()