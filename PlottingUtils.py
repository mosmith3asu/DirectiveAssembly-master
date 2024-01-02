# Imports ######################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
# Main #########################
def main():
    """ Main Function Call """





# Functions ####################
def simple_array_plot(A,label_colors =None):
    figure, ax = plt.subplots()
    labels = np.unique(A)
    unfilled_val = 0
    boundry_val = -1
    if label_colors is None:
        c = ['orange', 'blue', 'black', 'pink', 'yellow', 'green']
        label_colors = [random.choice(c) for _ in labels]
        label_colors[np.where(labels == unfilled_val)] = 'white'
        label_colors[np.where(labels == boundry_val)] = 'black'
    else:
        # for i in np.where(np.array(label_colors)=='black')[0]:
        #     label_colors[i]='grey' # change black to grey
        label_colors = ['grey','white']+label_colors
    tile_bounds = list(range(-1, np.size(label_colors) - 1))
    cmap = colors.ListedColormap(label_colors)
    norm = colors.BoundaryNorm(tile_bounds, cmap.N)
    ax.imshow(A, cmap=cmap, norm=norm)
    plt.show()





# Run ##########################
if __name__ == '__main__':
    main()
