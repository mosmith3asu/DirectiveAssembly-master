import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.BlockAssets import BlockDataClass,BlockSets
from SolutionSearch.utils.DataManagment import save_solutions
from SolutionSearch.utils.StructureAssets import FunnelObj,OvalObj,SquareObj
from SolutionSearch.utils.Visualization import plot_structure,plot_mask
from SolutionSearch.utils.Checks import check_unique_rotations,check_invalid_region,\
    check_out_of_bounds,check_overlap, check_is_solution,check_is_solution_mask,check_invalid_region_mask
import itertools
import math
import copy
import multiprocessing

n_workers = 9


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
# SOLUTIONS = {}

def iterate_blocks(structure,mask,blocks,block_states,SOLUTIONS,k=0,plot_sol=False):
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
                print(f'\r \t| [{round(100*SOLUTIONS[f"completed"]/SOLUTIONS[f"total"],2)}%][n={len(SOLUTIONS)}] found!', end='') #
                SOLUTIONS[len(SOLUTIONS)-2] = [copy.deepcopy(blk.data) for blk in blocks]
                if plot_sol: plot_structure(structure, blocks)
                this_mask = mask.copy() # remove block and continue to find solution
                break # if solution is found, this is only valid placement for block
            else:
                iterate_blocks(structure, this_mask, blocks, block_states,SOLUTIONS, k=k+1)

        # remove placed block  if invalid
        else: this_mask = mask.copy()
    blocks[k]((0,0,0))



def mp_worker(iworker,structure,my_combs,valid_states,SOLUTIONS):
    print(f'\t| Spawned worker {iworker}...')
    blocks = copy.deepcopy(structure.blocks)
    for ic, c in enumerate(my_combs):
        mask = structure.mask.copy()
        comb_states = [valid_states[b] for b in c]  # take only valid states for this combination of blocks
        comb_blocks = [blocks[b] for b in c]
        iterate_blocks(structure, mask, comb_blocks, comb_states,SOLUTIONS, k=0)
        SOLUTIONS[f"completed"] += 1
    print(f'\t| FINISHED worker {iworker}...')

def split_combs(combs):
    """
    Splits combinations of blocks into equal parts for workers to solve
    :param combs: list of valid checksum block combination IDs
    :return: list of comb subsets with len n_workers
    """
    n_per_split = math.ceil(len(combs)/n_workers)
    comb_slits = []
    for i in range(n_workers):
        imax = min((i+1)*n_per_split,len(combs))
        this_split = combs[i * n_per_split:imax]
        comb_slits.append(this_split)
    return comb_slits


def search_combination_solutions_heiarchical(structure,valid_combs,valid_states):
    """
    Takes combinations of block IDs and searches for perfect solutions
    :param structure:
    :param valid_combs:
    :return:
    """
    # valid_combs = valid_combs[:16]

    print(f'\n### Searching Possible Combinations for Solutions ###')
    print(f'\t| Structure: [{structure.name}]')

    # Split combinations for each worker -------------------------------
    comb_slits = split_combs(valid_combs)

    # Set up MP manager ------------------------------------------------
    manager = multiprocessing.Manager()
    SOLUTIONS = manager.dict()
    SOLUTIONS[f"total"] = len(valid_combs)
    SOLUTIONS[f"completed"] = 0
    jobs = []

    # Start jobs ------------------------------------------------
    for i in range(n_workers):
        p = multiprocessing.Process(target=mp_worker, args=(i, structure,comb_slits[i],valid_states,SOLUTIONS))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    # Save ------------------------------------------------
    SOLUTIONS = SOLUTIONS.copy() # convert from MP.manager DictProxy to dict
    del SOLUTIONS['total']
    del SOLUTIONS['completed']
    if len(SOLUTIONS) > 0:
        SOLUTIONS['structure'] = structure.name
        SOLUTIONS['blocklist'] = structure.block_list
        save_solutions(structure.name, SOLUTIONS)
    else:
        print(f'NO SOLUTIONS FOUND')




def main():
    # structure = FunnelObj()
    structure = OvalObj()
    # structure = SquareObj()
    valid_combs = get_checksum_combinations(structure)
    valid_states = search_valid_states(structure)
    search_combination_solutions_heiarchical(structure,valid_combs,valid_states)



if __name__ == "__main__":
    main()
