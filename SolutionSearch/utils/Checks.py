import numpy as np
import scipy.ndimage
# import matplotlib.pyplot as plt


def main():
    from SolutionSearch.utils.Visualization import plot_structure
    from SolutionSearch.utils.StructureAssets import FunnelObj

    Funnel = FunnelObj()
    blocks = Funnel.blocks[0:2]
    blocks[0]((4, 1, 0))
    blocks[1]((3, 3, 0))
    # blocks[2]((2, 4, 2))
    # blocks[3]((4, 3, 3))
    # blocks[4]((1, 6, 1))
    # blocks[5]((4, 5, 1))
    # blocks[6]((6, 5, 2))

    has_valid_regions = check_invalid_region(Funnel, blocks)
    print(has_valid_regions)
    plot_structure(Funnel, blocks)

    # a =np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #      [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    #      [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    #      [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#######################################################################
def check_is_solution(structure,blocks):
    if not isinstance(blocks, list): blocks = [blocks]
    # Place blocks------------
    check_mask = structure.mask.copy()
    for blk in blocks:
        check_mask += -1 * blk.mask
    return np.all(check_mask==0)

def check_is_solution_mask(check_mask):
    return np.all(check_mask==0)

#######################################################################
def check_invalid_region(structure,blocks, min_area=3, get='mask'):
    """
    Check if any regions are below minum require area of available blocks
    :param structure:
    :param blocks:
    :return:
    """
    # Input checks------------
    assert get == 'bool'
    if not isinstance(blocks, list): blocks = [blocks]
    mask_sz = blocks[0].mask.shape

    # Place blocks------------
    check_mask = structure.mask.copy()
    for blk in blocks:
        check_mask += -1 * blk.mask

    # Remove out of bound regions ---------
    check_mask[np.where(check_mask<0)]=0
    # Place blocks------------
    regions, num_features = scipy.ndimage.label(check_mask)
    # print(regions)
    region_areas = [len(np.where(regions == label)[0]) for label in np.unique(regions)]
    # print(region_areas)
    has_invalid_region = np.any(np.array(region_areas) < min_area)
    return has_invalid_region


def check_invalid_region_mask(check_mask, min_area=3, get='mask'):
    """
    Check if any regions are below minum require area of available blocks
    :param structure:
    :param blocks:
    :return:
    """
    assert get == 'bool'

    # Remove out of bound regions ---------
    check_mask[np.where(check_mask<0)]=0
    # Place blocks------------
    regions, num_features = scipy.ndimage.label(check_mask)
    region_areas = [len(np.where(regions == label)[0]) for label in np.unique(regions)]
    has_invalid_region = np.any(np.array(region_areas) < min_area)
    return has_invalid_region

def detect_region_shapes(WS):
    # Assign unique values over all similar regions
    regions = scipy.ndimage.label(WS)[0]
    labels = np.unique(regions)
    shapes = []
    for label in labels:
        region = regions
        del_r = np.all(regions != label, axis=1)
        region = np.delete(region, del_r, 0)
        del_c = np.all(regions != label, axis=0)
        region = np.delete(region, del_c, 1)
        shapes.append(np.shape(region))
    return shapes

def check_unique_rotations(blk):
    """
    Checks if any rotations of block are identical
    :param block:
    :return:
    """
    sym_list = ['U_2x2','I_3x1','I_3x2','Z_3x3','Z_3x2','H_3x3']
    if blk.name in sym_list:
        return [0,1]
    else: return [0,1,2,3]

    # x0,y0,r0 = blk.state
    # # blk = block.copy.deepcopy()
    # mask_list = []
    # for r in range(4):
    #     blk((0,0,r))
    #     # bmask = blk.mask.copy()
    #     # bmask = bmask[:3, :3]
    #     # bmask = bmask[~np.all(bmask == 0, axis=0)]
    #     # # bmask = bmask[~np.all(bmask == 0, axis=1)]
    #     #
    #     # mask_list.append(bmask)
    #     mask_list.append(blk.mask.copy())
    #
    # # Check if any masks are identical
    # unique_rots = [None,None,None,None]
    # # unique_rots = []
    # for m,mask in enumerate(mask_list):
    #     compare_lst = copy.copy(mask_list[:m])
    #     # compare_lst.pop(m)
    #
    #     has_duplicates = []
    #     for cmp_mask in compare_lst:
    #         has_duplicates.append(np.all(cmp_mask==mask))
    #
    #     # if len(has_duplicates)>0  and not np.any(has_duplicates):
    #     #     unique_rots.append(m)
    #     unique_rots[m] = True if len(has_duplicates)==0  else not np.any(has_duplicates)
    #
    # blk((x0, y0, r0)) # return block to origonal loc
    # return list(np.where(unique_rots)[0])




def check_out_of_bounds(structure, blocks, get='mask'):
    """checks any overlap in series of blocks
    get: {mask,loc,bool}
    """
    # Input checks------------
    assert get in ['mask', 'loc', 'bool']
    if not isinstance(blocks, list):
        blocks = [blocks]
    mask_sz = blocks[0].mask.shape

    # Place blocks------------
    check_mask = structure.mask.copy()
    for blk in blocks:
        check_mask += -1*blk.mask

    # Dynamic return------------
    if get == 'mask':
        oob_mask = np.zeros(mask_sz)
        oob_mask[np.where(check_mask <= -1)] = -1
        return oob_mask
    elif get == 'loc':
        return np.where(check_mask <= -1)
    elif get == 'bool':
        return np.any(check_mask <= -1)
    else:
        raise Exception('Unknown return error')


def check_overlap(blocks,get='mask'):
    """checks any overlap in series of blocks
    get: {mask,loc,bool}
    """
    # Input checks------------
    assert get in ['mask','loc','bool']
    mask_sz = blocks[0].mask.shape

    # Place blocks------------
    check_mask = np.zeros(mask_sz)
    for blk in blocks:
        check_mask += blk.mask


    # Dynamic return ------------
    if get == 'mask':
        overlap_mask = np.zeros(mask_sz)
        overlap_mask[np.where(check_mask > 1)] = -1
        return overlap_mask
    elif get == 'loc':
        return np.where(check_mask > 1)
    elif get == 'bool':
        return np.any(check_mask > 1)
    else:
        raise Exception('Unknown return error')





if __name__ == "__main__":
    main()
