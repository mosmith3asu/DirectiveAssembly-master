# import numpy as np
# import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from SolutionSearch.utils.StructureAssets import FunnelObj,OvalObj,SquareObj


def add_minimum_blocklist(fname):
    """
    Searches solution sets and finds the minimum number of blocks that describe the solution set.
    Removes extrenous blocks if any.
    :param fname:
    :return:
    """
    pass

def load_solutions(data_name,dir = 'Solutions/'):
    print(f'Loading solutions...')
    fname = f'{dir}{data_name}'
    print(f'\t| fname    = {fname}')
    with open(fname, "rb") as input_file:
        sols = pickle.load(input_file)

    structure_name = sols['structure']
    block_list = sols['blocklist']
    del sols['structure']
    del sols['blocklist']

    if structure_name == 'funnel':  structure = FunnelObj(block_list=block_list)
    elif structure_name == 'oval':  structure = OvalObj(block_list=block_list)
    elif structure_name == 'square': structure = OvalObj(block_list=block_list)
    else: raise Exception(f'UNKNOWN STRUCTURE NAME {structure_name}')
    print(f'\t| N blocks = {len(block_list)}')
    print(f'\t| N sols   = {len(sols)}')
    print(f'\t| finished...')
    return structure, sols




def save_solutions(struct_name, sols,dir = 'Solutions/'):
    print(f'Saving....')

    now = datetime.now()  # current date and time
    date_str = now.strftime('%m-%d-%Y')
    time_str = now.strftime('%H-%M-%S')
    fname = now.strftime(f"{dir}{struct_name}__N{len(sols)}__D{date_str}__T{time_str}.pkl")
    print(f'\t| fname:{fname}')
    with open(fname, 'wb') as handle:
        pickle.dump(sols, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    dir = '../Solutions/'
    data_name = 'funnel__N137__D03-23-2024__T19-32-52.pkl'
    structure, sols = load_solutions(data_name,dir=dir)

    # # load_solutions(r'C:\PycharmProjects\DirectiveAssembly-master\SolutionSearch\Solutions\funnel__N528__D03-22-2024__T16-50-01.pkl')

    # save_solutions('testsave',test_dict,dir=dir)
    # with open(f'{dir}funnel__N528__D03-22-2024__T16-50-01.pkl', 'rb') as handle:
    #     b = pickle.load(handle)

    #
    # with open(f'{dir}{fname}', 'rb') as handle:
    #     b = pickle.load(handle)

    print(sols)
