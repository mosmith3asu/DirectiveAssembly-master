import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from SolutionSearch.utils.Checks import check_overlap,check_out_of_bounds,detect_region_shapes,check_is_solution

def plot_structure(structure,blocks, ax=None, show=True):
    """
    Shows assembly withe errors in red
    :param structure: assembly strucuture
    :param blocks:  list of blocks
    :param ax:  optional axis, make new ax if none
    :param show: show the plot or pass
    :return: None
    """
    Clist = ['white', 'lightgray', 'green', 'blue', 'brown', 'olive', 'pink', 'cyan','darkorchid','darkblue','wheat']
    assert len(blocks)<len(Clist)
    smask = structure.mask.copy()

    # Check out of bounds ----------
    for blk in blocks:
        oob_loc = check_out_of_bounds(structure,blk,get='loc')
        smask += blk.mask* (blk.ID+1) # add block to workspace
        smask[oob_loc] = - 1 # set out of bounds elements

    # Check overlaps ----------
    overlaps = check_overlap(blocks, get='loc')
    smask[overlaps] = -1

    # print(smask)
    # smask[np.where(smask<=-1)] = 0



    # Fix colormap for invalid placmeents ----------
    if np.any(smask==-1):
        smask += 1
        Clist = ['red'] + Clist
    ncolors = len(np.unique(smask))
    cmap = ListedColormap(Clist[:ncolors])

    # Plot array as image -----------------
    if ax is None: fig,ax = plt.subplots(1,1)
    ax.imshow(smask, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if show: plt.show()

def plot_mask(mask, ax=None, show=True):
    """
    Shows assembly withe errors in red
    :param structure: assembly strucuture
    :param blocks:  list of blocks
    :param ax:  optional axis, make new ax if none
    :param show: show the plot or pass
    :return: None
    """
    Clist = ['white', 'lightgray']


    # Fix colormap for invalid placmeents ----------
    if np.any(mask==-1):
        mask += 1
        Clist = ['red'] + Clist
    cmap = ListedColormap(Clist)

    # Plot array as image -----------------
    if ax is None: fig,ax = plt.subplots(1,1)
    ax.imshow(mask, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if show: plt.show()



def main():
    from SolutionSearch.utils.StructureAssets import FunnelObj
    Funnel = FunnelObj()
    blocks = Funnel.blocks
    blocks[0]((4, 2, 0))
    blocks[1]((3, 3, 0))
    blocks[2]((2, 4, 2))
    blocks[3]((4, 3, 3))
    blocks[4]((1, 6, 1))
    blocks[5]((4, 5, 1))
    blocks[6]((6, 5, 2))
    print(check_is_solution(Funnel,blocks))

    plot_structure(Funnel, blocks)


if __name__ == "__main__":
    main()
