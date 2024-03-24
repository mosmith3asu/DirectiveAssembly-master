import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.DataManagment import load_solutions
from SolutionSearch.utils.BlockAssets import BlockDataClass
import copy

class CoactiveAgent():
    """
    IF AVAILABLE SOLUTIONS:
        Place solution that contains the largest solution reward
        Place solution that maximizes # of available solutions
        Place block closest to top
        Place block closest to left
    ELSE:
        Remove solution s.t. new solutions contain the largest possible reward
        Remove solution s.t. number of new solutions are maximized
        Remove block closest to top
        Remove block closest to left
    """
    def __init__(self,structure, solutions):
        self.agent_type = 'coactive'
        print(f'\nInitializing {self.agent_type} agent...')
        self.null_state = np.array([0,0,0]) # designates out of play block
        self.structure = structure         # origonal structure obj
        self.ws_sz = structure.mask.shape  # size of workspace
        self.n_solutions = len(solutions)   # original # of solutions
        self.n_blocks = len(structure.block_list)
        self.solutions = self.solution_list2dict(solutions) # dict is better for keeping track of solution index when deleting
        self.block_names = structure.block_list # avaliable blocks in blockpool
        self._ws_mask0 = self.structure.mask.copy()  # blank workspace mask

        # Persistent vars
        print(f'\t| Precomputing metrics...')
        self.all_solution_states = self.calc_solution_states()
        # self.all_solution_block_IDs = self.calc_solution_IDs()
        self.all_solution_rewards = self.calc_solution_rewards()
        # self.all_solution_ID_masks = self.calc_solution_ID_masks()

        # Dynamic vars
        print(f'\t| Initializing workspace...')
        self.blocks = self.spawn_blocks_obj_dict(self.block_names)
        self.unfilled_mask = structure.mask.copy()     # current workspace
        # self.current_solutions = copy.deepcopy(solutions) # current list of solutions
        # self.current_solution_IDS = (np.arange(self.n_solutions))  # current list of solutions
        self.current_solution_states = copy.deepcopy(self.all_solution_states) # current list of solutions states
        self.placed_block_IDs = []                  # blocks currently paced in workspace
        self.unplaced_block_IDs = [ID for ID in range(self.n_blocks)]  # blocks out of the workspace (blockpool)

        print(f'\t| Finished')

    ########################################################################
    ## Precompute Metrics ##################################################
    ########################################################################
    def solution_list2dict(self,solutions):
        sol_dict = {}
        for isol in range(self.n_solutions):
            sol_dict[isol] = solutions[isol]
        return sol_dict

    def spawn_blocks_obj_dict(self,blocklist):
        block_dict = {}
        for block_ID,block_name in enumerate(blocklist):
            # block_dict[iblock] = self.get_block(block)
            block_dict[block_ID] = BlockDataClass(block_ID, block_name, self.null_state)
        return block_dict

    def calc_solution_rewards(self,as_array=True):
        """
        Calculate list of reward recieved for each of the defined solutions
        :return: dict of int rewards of len=len(self.solutions)
        """

        print(f'\t|\t| Generating solution rewards...')
        block_counts = np.zeros(self.n_solutions)
        for isol in range(self.n_solutions):
            blocks = self.solutions[isol]
            block_counts[isol] = len(blocks)
        sol_rewards = np.max(block_counts) - block_counts  # invert s.t. lower counts have higher reward
        if as_array: return sol_rewards

        # convert to dict
        sol_reward_dict = {}
        for isol in range(self.n_solutions):
            sol_reward_dict[isol] = sol_rewards[isol]
        return sol_reward_dict


    def calc_solution_IDs(self):
        """
        DEPRICATED
        Calculate list of block IDs in each solution
        :return: int rewards of len=len(self.solutions)
        """
        print(f'\t|\t| Generating list of block IDs in each solution...')
        sol_IDs = {}
        for isol in range(self.n_solutions):
            blocks = self.solutions[isol]
            ID_dict = {}
            for block in blocks:
                ID_dict[block['ID']] = block['ID']
                # ID_list.append(block['ID'])
            sol_IDs[isol] = ID_dict
        return sol_IDs

    def calc_solution_states(self):
        """
        Calculate list of block states in each solution
        :return: int rewards of len=len(self.solutions)
        """
        print(f'\t|\t| Generating list of block IDs in each solution...')
        sol_states = {}
        for isol in range(self.n_solutions):
            blocks = self.solutions[isol]
            states_dict = {} # dict of current block ID
            for block in blocks:
                states_dict[block['ID']] = block['state']
            sol_states[isol] = states_dict
        return sol_states

    def calc_solution_ID_masks(self):
        """
        Generate masks of all possible solutions with block ID values in mask
        """
        print(f'\t|\t| Generating solution masks...')
        sol_masks =  {} #np.zeros([self.n_solutions,self.ws_sz[0],self.ws_sz[1]])
        for isol in range(self.n_solutions):
            blocks = self.solutions[isol]
            mask = np.zeros([self.ws_sz[0],self.ws_sz[1]])
            for block in blocks:
                mask += self.get_block(block).mask* (block['ID']+1)
            sol_masks[isol] = mask
        return sol_masks


    ########################################################################
    ## Action Selection ####################################################
    ########################################################################
    def pick(self):
        """
        Executes high-level decision-making of other pick actions in following priority: reward -> rem_sol ->location
        :return: block placement choice
        """
        if len(self.current_solution_states)>0:
            sol_IDs = list(self.current_solution_states.keys())
            sol_IDs = self.pick_solutions_with_max_reward(sol_IDs)
            block_choices = self.pick_blocks_with_max_rem_sol(sol_IDs)
            block_ID,state = self.pick_block_by_location(block_choices)

            rem_sol = self.check_remaining_solutions(block_ID,state)
            print(f'Block {block_ID} with state {state} chosen with n={len(rem_sol)} remaining solutions')
            return block_ID,state

        else: # remove block
            assert Exception('Unfinished...')
    def pick_solutions_with_max_reward(self,sol_IDs):
        """
        Pick the block placement(s) that generate solutions with the highest potential reward
        :return: list of block placements (len=1 if only one choice remains)
        """
        imax_reward = np.where(self.all_solution_rewards[sol_IDs]==np.max(self.all_solution_rewards[sol_IDs]))
        return  np.array(sol_IDs)[imax_reward].tolist()  # return only solutions with max reward
    def pick_blocks_with_max_rem_sol(self,sol_IDs):
        """
        Pick the block placement(s) that result in solutions with the highest potential reward
        :return: list of block placements (len=1 if only one choice remains)
        """
        checked_hash = {}
        for sol_ID in sol_IDs:
            for block_ID in self.unplaced_block_IDs:
                # Check if this block is in this solution set
                sol_block_IDs = list(self.current_solution_states[sol_ID].keys())
                if block_ID in sol_block_IDs:
                    state = self.current_solution_states[sol_ID][block_ID]
                    # See if we checked block before to avoid expensive search for rem solutions
                    hashID = (block_ID,state[0],state[1],state[2]) #f'{block_ID}{state}'
                    if not hashID in checked_hash:
                        rem_sols = self.check_remaining_solutions(block_ID,state)
                        checked_hash[hashID] = len(rem_sols)

        # Get the keys in checked_hash that have the max number of remaining solutions
        max_rem_sol = max(checked_hash.values())
        block_placements = [hashID for hashID in checked_hash if checked_hash[hashID] == max_rem_sol]
        return block_placements
    def pick_block_by_location(self, placements):
        """
        picks the next block to place based on location (top left first)
        :return: single block placement
        """
        placements = np.array(placements)

        # Choose block with smallest y value
        miny = np.min(placements[:,1])
        iplace = np.where(placements[:,1]==miny)[0] # get the block with the smallest x value

        # Choose block with smallest x value
        if len(iplace)>1: # same y vals
            minx = np.min(placements[:, 2])
            iplace = np.where(placements[:, 2] == minx)[0]

        # Randomly sample if same
        if len(iplace) > 1:  # same y and x vals
            iplace = [np.random.choice(iplace)]

        # Return block ID and state
        iplace = iplace[0]
        block_ID = placements[iplace,0]
        state = placements[iplace,1:]
        return  block_ID,state.tolist()

    ########################################################################
    ## Evaluation of Gamestate #############################################
    ########################################################################
    def get_max_potential_reward(self,placed_blocks):
        """
        Gets the maximum potential reward of placed blocks in the WS given available blocks
        :param placed_blocks:
        :return:
        """

    def check_remaining_solutions(self, block_ID,state,to_remove=False):
        """
        Get the remaining solutions given the placed blocks
        - Option 1: check if any remaining blocks are a subset of each self.solutions
        - Option 2: Somehow check the mask? (faster?)
        - Option 3: Do recursive search (likely slower depending on how many blocks remaining)

         # valid solutions must match states of placed blocks (eliminates other block placements)
            # invalid solutions do not contain block ID from placed blocks
            # invalid solutions do not contain block ID in state S from placed blocks.
            # Always order in increasing ID order to decrease compare time
            # REDUNDANT (DONT DO): valid solutions must contain IDs of placed block  (eliminates other block combinations)
            # REDUNDANT (DONT DO): valid solutions must contain IDs of unplaced blocks
        :return:
        """
        sol2remove = []
        rem_sol_IDs = []  # lists of invalidated indices
        for sol_ID, sol_block_states in self.current_solution_states.items():
            sol_block_IDs = list(sol_block_states.keys())

            # invalid solution ==> remove; do not search blocks
            is_valid = True
            if not block_ID in sol_block_IDs: is_valid = False; sol2remove.append(sol_ID)
            elif not np.all(state == sol_block_states[block_ID]): is_valid = False; sol2remove.append(sol_ID)
            if is_valid: rem_sol_IDs.append(sol_ID)

        # print(f'Nsol = {len(self.current_solution_states)}')
        if to_remove: return sol2remove # return which solutions to remove
        else: return rem_sol_IDs # return which solutions are valid

    # def recalc_all_remaining_solutions(self):
    #     """
    #     Get the remaining solutions given the placed blocks
    #     - Option 1: check if any remaining blocks are a subset of each self.solutions
    #     - Option 2: Somehow check the mask? (faster?)
    #     - Option 3: Do recursive search (likely slower depending on how many blocks remaining)
    #
    #      # valid solutions must match states of placed blocks (eliminates other block placements)
    #         # invalid solutions do not contain block ID from placed blocks
    #         # invalid solutions do not contain block ID in state S from placed blocks.
    #         # Always order in increasing ID order to decrease compare time
    #         # REDUNDANT (DONT DO): valid solutions must contain IDs of placed block  (eliminates other block combinations)
    #         # REDUNDANT (DONT DO): valid solutions must contain IDs of unplaced blocks
    #     :return:
    #     """
    #     sol2remove = [] # lists of invalidated indicies
    #     for sol_ID, sol_block_states in self.current_solution_states.items():
    #         sol_block_IDs = list(sol_block_states.keys())
    #
    #         for block_ID in self.placed_block_IDs:
    #             # invalid solution ==> remove; do not search blocks
    #             if not block_ID in sol_block_IDs: sol2remove.append(sol_ID);  break
    #             elif not np.all(self.blocks[block_ID].state == sol_block_states[block_ID]): sol2remove.append(sol_ID); break
    #
    #     for sol_ID in sol2remove:
    #         del self.current_solution_states[sol_ID]
    #
    #     print(f'Nsol = {len(self.current_solution_states)}')
    #     return None

    ########################################################################
    ## Gamestate Updates ###################################################
    ########################################################################

    def update_place_block(self,block_ID,state):
        state = np.array(state)
        self.blocks[block_ID](state) # update block obj
        self.placed_block_IDs.append(block_ID)  # add to placed blocks
        self.placed_block_IDs.sort()  # Make sure lists are sorted
        self.unplaced_block_IDs.remove(block_ID) # remove from unplaced blokcs
        self.unfilled_mask += -1*self.blocks[block_ID].mask # update unfilled mask
        # self.recalc_remaining_solutions() # update feasible solutions
        sol2remove = self.check_remaining_solutions(block_ID,state, to_remove=True)
        for sol_ID in sol2remove:
            del self.current_solution_states[sol_ID]
        print(f'Remaining solutions = {len(self.current_solution_states)}')

    # def update_remove_block(self,ID):
    #     self.blocks[ID](self.null_state)  # update block obj
    #     self.unplaced_block_IDs.append(ID)  # add to placed blocks
    #     self.unplaced_block_IDs.sort()  # Make sure lists are sorted
    #     self.placed_block_IDs.remove(ID)  # remove from unplaced blokcs
    #     self.unfilled_mask += self.blocks[ID].mask  # update unfilled mask
    #     self.recalc_all_remaining_solutions()  # update feasible solutions





    ########################################################################
    ## Utilities ###########################################################
    ########################################################################
    # def place_block(self,ws_mask,block):
    #     return ws_mask +

    def get_block(self,block):
        return BlockDataClass(block['ID'], block['name'], block['state'])

    @property
    def blank_workspace(self):
        return self._ws_mask0.copy()






# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    directory = '../SolutionSearch/Solutions/'
    # data_name = 'funnel__N137__D03-23-2024__T20-03-05.pkl'
    data_name = 'funnel__N2167__D03-23-2024__T21-47-08.pkl'
    structure, sols = load_solutions(data_name, dir=directory)

    agent = CoactiveAgent(structure, sols)
    # agent.update_place_block(block_ID=0, state=(1, 0, 1))
    agent.update_place_block(block_ID=0,state=(3,2,1)) # in solution 0
    for ib in range(7):
        block_ID, state = agent.pick()
        agent.update_place_block(block_ID=block_ID, state=state)
        print(agent.unfilled_mask)




if __name__ == "__main__":
    main()
