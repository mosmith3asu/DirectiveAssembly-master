import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.BlockAssets import BlockDataClass,BlockSets
from SolutionSearch.utils.StructureAssets import FunnelObj
from SolutionSearch.utils.Visualization import plot_structure,plot_mask
from SolutionSearch.utils.Checks import check_unique_rotations,check_invalid_region,\
    check_out_of_bounds,check_overlap, check_is_solution,check_is_solution_mask,check_invalid_region_mask
import itertools
import math
import random
import copy
import pickle

def get_checksum_combinations(structure):
    """
    finds all possible combinations of blocks whose cumulative area perfectly fills the structure.
    :param structure:
    :return:
    """
    print(f'\n### Checksum Combinations ###')
    print(f'\t| Structure: [{structure.name}]')
    min_number_of_blocks = 2
    N = structure.n_blocks
    ID_list = np.arange(N)
    valid_combs = []
    blocks = structure.blocks
    for r in range(min_number_of_blocks,N+1):
        # get possible combinations of length r
        combs = itertools.combinations(ID_list, r)

        # check if combinations pass checksum
        for c in combs:
            comb_sum = np.sum([blocks[ID].sum for ID in c])
            if structure.sum == comb_sum:
                valid_combs.append(c)
                # print('Found valid checksum...')
    print(f'\t| Found {len(valid_combs)} valid checksum combinations...')
    return valid_combs

def search_valid_states(structure):
    print(f'\n### Searching Valid Block States ###')
    print(f'\t| Structure: [{structure.name}]')

    poss_x = list(np.arange(1, 8)) # possible x vals
    poss_y = list(np.arange(1, 8)) # possible y vals
    blocks = structure.blocks

    # Reduce number of states for each block by invalidating placements --------------------
    valid_states = []
    for block_ID, blk in enumerate(blocks):
        # Get possible states for this block
        poss_r = check_unique_rotations(blk)  # possible r vals
        # print(f'{blk.ID} rots: {poss_r}')
        poss_states = itertools.product(*[poss_x, poss_y, poss_r])  # permutation of possible states

        # Loop through placements
        this_valid_states = []
        for state_ID, state in enumerate(poss_states):
            blk(state)  # move to new state
            has_invalid_region = check_invalid_region(structure,blk,get='bool')
            has_out_of_bounds = check_out_of_bounds(structure,blk,get='bool')
            if not has_out_of_bounds and not has_invalid_region:
                this_valid_states.append(state)
        print(f'\t| [block {block_ID}] has [{len(this_valid_states)}] valid states')
        valid_states.append(this_valid_states)
    return valid_states


############################################################
SOLUTIONS = {}

def iterate_blocks(structure,mask,blocks,block_states,k,plot_sol=False):
    # SOLS: [9,3,1,34,0,23,29]

    this_block =  blocks[k]
    this_mask = mask.copy()


    for istate, bstate in enumerate(block_states[k]):

        # place block
        this_block(bstate)
        this_mask -= this_block.mask

        # Check validity
        has_overlap = np.any(this_mask<0) # checks for overlaps and out of bounds
        not_feasible = check_invalid_region_mask(this_mask.copy(),get='bool') # checks for min area (forward thinking)
        is_valid = (not has_overlap and not not_feasible)

        # Save solution or increase recursion depth
        if is_valid:
            if k == len(blocks) - 1:
                print(f'\r [n={len(SOLUTIONS)}] found!', end='')
                # print([blk.state for blk in blocks])
                # SOLUTIONS.append([copy.deepcopy(blk.data) for blk in blocks])
                SOLUTIONS[len(SOLUTIONS)+1] = [copy.deepcopy(blk.data) for blk in blocks]
                if plot_sol: plot_structure(structure, blocks)
            else:
                iterate_blocks(structure, this_mask, blocks, block_states, k + 1)

        # remove placed block  if invalid
        else: this_mask = mask.copy()
    blocks[k]((0,0,0))


def search_combination_solutions_heiarchical(structure,valid_combs,valid_states):
    """
    Takes combinations of block IDs and searches for perfect solutions
    :param structure:
    :param combs:
    :return:
    """

    print(f'\n### Searching Possible Combinations for Solutions ###')
    print(f'\t| Structure: [{structure.name}]')
    valid_states_ID = [list(np.arange(len(blk_states))) for blk_states in valid_states]
    blocks = structure.blocks

    n_combs=len(valid_combs)
    for ic, c in enumerate(valid_combs):
        print(f'\nStarting Comb: {ic}/{n_combs} | n_blocks= { len(c)}')
        print(f'\r [n={len(SOLUTIONS)}] found!', end='')

        mask = structure.mask.copy()
        comb_states =  [valid_states[b] for b in c] # take only valid states for this combination of blocks
        comb_blocks = [blocks[b] for b in c]
        iterate_blocks(structure,mask, comb_blocks, comb_states, k=0)
    if len(SOLUTIONS)>0: save_solutions(structure.name,SOLUTIONS)


def main():
    Funnel = FunnelObj()



    valid_combs = get_checksum_combinations(Funnel)
    valid_states = search_valid_states(Funnel)
    # search_combination_solutions(Funnel,valid_combs,valid_states)
    # search_combination_solutions_rand(Funnel,valid_combs,valid_states)

    search_combination_solutions_heiarchical(Funnel,valid_combs,valid_states)


    # b = BlockDataClass(ID=1, name="L_2x2", state=[1, 2, 4], color='r')
    # print(b)
    # print(b.data)
    # new_state = [0, 0, 0]
    # print(f'{b[:]}=>{b * 2} and {b * [1, 3]}= {b * [1, 3, 5]}')
    # print(f'{b(new_state)}')
    # # print(b.layer)
    # # print(b.sum)
    # print(Funnel.mask)
    # i=2
    # print(Funnel.n_blocks)
    # print(Funnel.blocks[i].name)
    # print(Funnel.blocks[i].sum)
    # print(Funnel.blocks[i].mask)



def save_solutions(struct_name, sols):
    from datetime import datetime
    now = datetime.now()  # current date and time
    fname = now.strftime(f"Solutions/{struct_name}__N{len(sols)}__D%m-%d-%Y__T%H-%M-%S.pkl")
    with open(fname, 'wb') as handle:
        pickle.dump(sols, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
