import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.BlockAssets import BlockDataClass

def get_valid_block_placements(structure,block):
    pass


def main():
    b = BlockDataClass(ID=1, name="L_2x2", state=[1, 2, 4], color='r')
    print(b)
    print(b.data)
    new_state = [0, 0, 0]
    print(f'{b[:]}=>{b * 2} and {b * [1, 3]}= {b * [1, 3, 5]}')
    print(f'{b(new_state)}')



def subfun():
    pass


if __name__ == "__main__":
    main()
