# import numpy as np
# import matplotlib.pyplot as plt
import pickle
from SolutionSearch.utils.StructureAssets import FunnelObj,OvalObj,SquareObj


def add_minimum_blocklist(fname):
    """
    Searches solution sets and finds the minimum number of blocks that describe the solution set.
    Removes extrenous blocks if any.
    :param fname:
    :return:
    """
    pass

def load_solutions(fname):
    print(f'Loading solutions...')
    print(f'\t| fname    = {fname}')
    with open(fname, "rb") as input_file:
        sols = pickle.load(input_file)

    structure_name = sols['structure']
    block_list = sols['block_list']
    del sols['structure']
    del sols['block_list']

    if structure_name == 'funnel':  structure = FunnelObj(block_list=block_list)
    elif structure_name == 'oval':  structure = OvalObj(block_list=block_list)
    elif structure_name == 'square': structure = OvalObj(block_list=block_list)
    else: raise Exception(f'UNKNOWN STRUCTURE NAME {structure_name}')
    print(f'\t| N blocks = {len(block_list)}')
    print(f'\t| N sols   = {len(sols)}')
    print(f'\t| finished...')
    return structure, sols




def save_solutions(struct_name, sols):
    print(f'Saving....')
    from datetime import datetime
    now = datetime.now()  # current date and time
    fname = now.strftime(f"Solutions/{struct_name}__N{len(sols)}__D%m-%d-%Y__T%H-%M-%S.pkl")
    with open(fname, 'wb') as handle:
        pickle.dump(sols, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    fname = ''
    load_solutions()
