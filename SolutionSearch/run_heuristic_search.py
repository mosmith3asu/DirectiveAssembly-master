import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.BlockAssets import BlockDataClass,BlockSets
from SolutionSearch.utils.StructureAssets import FunnelObj
import itertools
import math
def get_checksum_combinations(structure):
    """
    finds all possible combinations of blocks whose cumulative area perfectly fills the structure.
    :param structure:
    :return:
    """
    min_number_of_blocks = 4
    N = structure.n_blocks
    ID_list = np.arange(N)
    valid_combs = []
    blocks = structure.blocks
    for r in range(min_number_of_blocks,N):
        # get possible combinations of length r
        combs = itertools.combinations(ID_list, r)

        # check if combinations pass checksum
        for c in combs:
            comb_sum = np.sum([blocks[ID].sum for ID in c])
            if structure.sum == comb_sum:
                valid_combs.append(c)
                # print('Found valid checksum...')
    print(f'[{structure.name}] Found {len(valid_combs)} valid checksum combinations...')
    return valid_combs

def search_combination_solutions(structure,combs):
    """
    Takes combinations of block IDs and searches for perfect solutions
    :param structure:
    :param combs:
    :return:
    """
    x_range = [1, 8]
    y_range = [1, 8]
    r_range = [0, 4]
    state_ranges = [list(np.arange(*x_range)),list(np.arange(*y_range)),list(np.arange(*r_range))]
    possible_states = list(itertools.product(*state_ranges))
    n_states = len(possible_states)
    state_IDs = list(np.arange(n_states))
    blocks = structure.blocks

    # Reduce number of states for each block by invalidating placements --------------------
    valid_state_IDs = []
    for block_ID, blk in enumerate(blocks):
        this_valid_states = []
        for state_ID,state in enumerate(possible_states):
            # print(blk.mask)
            blk(state) # move to new state
            bmask = blk.mask.copy()
            smask = structure.mask.copy()
            check_mask = smask - bmask
            if np.all(check_mask>=0): # not out of bounds
                this_valid_states.append(state_ID)
        print(f'[block {block_ID}] has {len(this_valid_states)} valid states')
        valid_state_IDs.append(this_valid_states)

    # Combine valid placemnts to test for solutions --------------------
    # print(f'# of possible states: {len(possible_states)}')

    for c in combs:
        n_blocks = len(c)

        # calc number of possible states
        valid_comb_state_IDs = [valid_state_IDs[block_ID] for block_ID in c]

        n_perms = np.product([math.factorial(len(lst)) for lst in valid_comb_state_IDs])
        print(n_perms)
        break
        # states_ID_combs = list(itertools.product(*valid_comb_state_IDs))
        # print(len(states_ID_combs))


        # this_mask = structure.mask.copy()
        # states_combs = list(itertools.product([possible_states for _ in range(n_blocks)]))
        # states_combs = list(itertools.product(*[state_IDs for _ in range(n_blocks)]))

        # for ID in c:
        #     for state in possible_states:
        #         blocks[ID](state)
        #         this_mask += blocks[ID]
        #


def main():
    Funnel = FunnelObj()
    checksum_combs = get_checksum_combinations(Funnel)
    search_combination_solutions(Funnel,checksum_combs)
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


def subfun():
    pass


if __name__ == "__main__":
    main()
