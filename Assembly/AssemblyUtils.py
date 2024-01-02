import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import sqrt, ceil
np.set_printoptions(linewidth=300) # removes array wraping in consol


def wait_key_press():
    input("Press Enter to continue...")

def append_buffer(Ai, labels=None, buff_size=0, buff_val=-2):
    """
    Append any list of arrays
    :param Ai: list of np.arrays to append together
    :param labels: location of labels if given
    :param buff_size: number of rows/cols used as buffer
    :param buff_val: value for buffer stored in returned array
    :return Acomb: combined array
    :return loc_labels: location of labels in plot {if given} assumes tile size=1,1
    """
    N = len(Ai)  # number of arrays

    if labels is None:
        enable_labels = False
    elif len(labels) != N:
        warning.warn("warn>>append_buffer>>Size of labels does not match size of Ai; ##disabling##")
        enable_labels = False
    else:
        enable_labels = True

    ###################################################################
    #### find bounding box that fits all arrays #######################
    bb_sz = [0, 0]  # bounding box size
    for i in range(N):
        h, w = np.shape(Ai[i])
        if bb_sz[0] < h: bb_sz[0] = h
        if bb_sz[1] < w: bb_sz[1] = w

    ####################################################################
    ### CREATE PADDED ARRAY OF SAME SHAPES #############################
    Abb = []  # list of Ai that had been padded to match shape
    for i in range(N):
        h, w = np.shape(Ai[i])
        dh, dw = np.array(bb_sz) - np.array([h, w])
        padh = int(dh / 2)
        padw = int(dw / 2)
        ptop = ceil(dh % 2)  # extra padding to take care of remainder
        pright = ceil(dw % 2)  # extra padding to take care of remainder
        Abb.append(np.pad(Ai[i], [(padh, padh + ptop), (padw, padw + pright)], mode='constant'))

    ##########################################################################
    ### ADD ALL ARRAYS THAT FIT IN SQUARE  ADD LEFTOVER ARRAYS ON BOTTOM  ####
    i = 0  # index to keep track of arrays added_
    hw = ceil(sqrt(N))  # shape of combined grid (excluding remainder)
    bb_fill = buff_val * np.ones(bb_sz)  # filler array for remainder

    # --- disabled ----
    # remaining = hw*hw % N # remaining arrays after square (remainder)
    remaining = 0  # remaining arrays after square (remainder)
    if remaining > 0:rextra = 1
    else:rextra = 0
    total_w = bb_sz[0] * hw + hw + 1
    total_h = bb_sz[1] * hw + hw + 1

    if buff_size == 0:
        cbuffer = buff_val * np.empty([bb_sz[1], buff_size])  # column buffer between arrays
        rbuffer = buff_val * np.empty([buff_size, bb_sz[0] * hw])  # row buffer between arrays
    else:
        cbuffer = buff_val * np.ones([bb_sz[1], buff_size])  # column buffer between arrays
        rbuffer = buff_val * np.ones([buff_size, bb_sz[0] * hw + hw + 1])  # row buffer between arrays

    LabelDict = {}
    bbbuffer_h = bb_sz[0] + buff_size
    bbbuffer_w = bb_sz[1] + buff_size

    Acomb = rbuffer
    for r in range(hw + rextra):
        addRow = cbuffer

        for c in range(hw):
            if i <= N - 1:
                if buff_size == 0:addRow = np.concatenate((addRow, Abb[i]), axis=1)
                else:addRow = np.concatenate((addRow, Abb[i], cbuffer), axis=1)
            else:
                if buff_size == 0:addRow = np.concatenate((addRow, bb_fill), axis=1)  # fill in after remaining
                else: addRow = np.concatenate((addRow, bb_fill, cbuffer), axis=1)  # fill in after remaining
            if enable_labels and i <= N - 1:
                label_loc = np.array([total_h + 1 - (r * bbbuffer_h + bbbuffer_h / 2),
                                      c * bbbuffer_w + bbbuffer_w / 2]) + 0.5  # label locations
                LabelDict[labels[i]] = label_loc
            i += 1
        if buff_size==0:
            Acomb = np.concatenate((Acomb, addRow), axis=0)
        else:Acomb = np.concatenate((Acomb, addRow, rbuffer), axis=0)

    if enable_labels:
        return Acomb, LabelDict
    else:
        return Acomb


def preview_grid(A, Title="", LabelDict=None, text_color='w', ColorDict=None):
    """
    Preview any array as plot
    :param A: Matrix to plot containing elements numbered to space's blocks/state
    :return:
    """
    if ColorDict is None:
        COLOR = {}
        COLOR[-10] = 'gainsboro'  # invalid, unfilled tile
        COLOR[-2] = 'gainsboro'  # invalid, unfilled tile
        COLOR[-1] = 'red'  # invalid, filled tile
        COLOR[0] = 'white'  # valid, unfilled tile
        COLOR[1] = 'green'  # valid, filled tile
        COLOR[2] = 'blue'  # valid, filled tile
        COLOR[3] = 'orange'  # valid, filled tile
        COLOR[4] = 'yellow'  # valid, filled tile
        COLOR[5] = 'cyan'  # valid, filled tile
        COLOR[6] = 'pink'  # valid, filled tile
        COLOR[7] = 'brown'  # valid, filled tile
        COLOR[8] = 'lime'  # valid, filled tile
        COLOR[9] = 'olive'  # valid, filled tile
    c_edge='lightgray'

    grid_h, grid_w = np.shape(A)  # height and width of plot
    tile_h, tile_w = (1, 1)  # block height & width in tiles

    fig, ax = plt.subplots()
    for r in range(grid_h):
        for c in range(grid_w):
            xy = (c, grid_h - r - 1)
            tile_val = A[r, c]
            fc = COLOR[tile_val]
            if fc=='white':
                rect = Rectangle(xy, tile_w, tile_h, linewidth=1, edgecolor='k', facecolor=fc)
            else:
                rect = Rectangle(xy, tile_w, tile_h, linewidth=1, edgecolor=fc, facecolor=fc)

            ax.add_patch(rect)

    if LabelDict is not None:
        for i, key in enumerate(LabelDict):
            label_loc = LabelDict[key]
            ax.text(label_loc[1], label_loc[0], key,
                    c=text_color, ha='center', va='center')

    plt.ylim(0, grid_h)
    plt.xlim(0, grid_w)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.title(Title)
    plt.axis('square')
    plt.show()
    #plt.show(block=False)
