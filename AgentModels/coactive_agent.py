import warnings

import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.DataManagment import load_solutions
from SolutionSearch.utils.BlockAssets import BlockDataClass
import copy
import itertools

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
        self.all_solution_rewards = self.calc_solution_rewards()
        # self.all_solution_ID_masks = self.calc_solution_ID_masks()

        # Dynamic vars
        print(f'\t| Initializing workspace...')
        self.blocks = self.spawn_blocks_obj_dict(self.block_names)
        self.unfilled_mask = structure.mask.copy()     # current workspace
        # self.block_ID_mask = np.zeros(self.unfilled_mask.shape) # current block ID mask (0=empty
        # self.current_solutions = copy.deepcopy(solutions) # current list of solutions
        # self.current_solution_IDS = (np.arange(self.n_solutions))  # current list of solutions
        self.current_solution_states = copy.deepcopy(self.all_solution_states) # current list of solutions states
        self.excluded_solution_states = {}                                      # solutions that are no longer valid
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
        ! DOES NOT PERFORM ACTION, SIMPLY SELECTS !
        :return: block placement choice <>ID,state>
        """
        if self.is_complete:
            warnings.warn('Agent tried to pick but assembly is finished...')
            return None, None

        # If there are still feasible solutions -----------------
        if len(self.current_solution_states)>0:
            sol_IDs = list(self.current_solution_states.keys())
            sol_IDs = self.pick_solutions_with_max_reward(sol_IDs)
            block_choices = self.pick_blocks_with_max_rem_sol(sol_IDs)
            block_ID,state = self.pick_block_by_location(block_choices)

            rem_sol = self.check_remaining_solutions(block_ID,state)
            # print(f'Block {block_ID} with state {state} chosen with n={len(rem_sol)} remaining solutions')
            return block_ID,state

        # Remove a block s.t. there are feasible solutions -----------------
        else: # remove block
            num_sol_after_remove = {}
            for block_ID in self.placed_block_IDs:
                state = self.blocks[block_ID].state
                rem_sol = self.check_remaining_solutions(block_ID,None)
                num_sol_after_remove[block_ID] = len(rem_sol)

            # find the keys where num_sol_after_remove values == max_num_sol
            max_num_sol = max(num_sol_after_remove.values())
            remove_options = [block_ID for block_ID in num_sol_after_remove.keys() if num_sol_after_remove[block_ID]==max_num_sol]
            block_ID = np.random.choice(remove_options)
            state = None
            # print(f'{num_sol_after_remove}')
            # print(f'Choosing to remove block {block_ID} with n={max_num_sol} remaining solutions')
            return block_ID,state


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

    def check_remaining_solutions(self, block_ID,state,to_remove=False,to_add=False,force_remove=False):
        """
        Check which solutions are still valid given the current block placements (or removals)
        :param block_ID: ID of desired block placement to check
        :param state: state of desired block placement to check (None if removing block)
        :param to_remove:  {only when adding block} flag to return which solutions to remove (True) or which solutions are still valid (False)
        :param to_add: {only when removing block} flag to return which solutions to add (True) or which solutions are still valid (False)
        :return:
        """
        # Check when removing block --------------------------------
        if (state is None or np.all(state==self.null_state)) or force_remove:
            """
            Valid solutions to add back in must...
                - Contain all of other placed blocks ID's
                - Have all of the other placed blocks in their current state
            """
            sol2add = []
            other_block_IDs = [ib for ib in self.placed_block_IDs if ib != block_ID]

            # No other placed blocks ==> all solutions are valid
            if len(other_block_IDs)==0:
                sol2add = list(self.excluded_solution_states.keys())

            else:
                # Check all excluded solutions
                for sol_ID, sol_block_states in self.excluded_solution_states.items():
                    sol_block_IDs = list(sol_block_states.keys())  # get IDs for this solution

                    # Check if all of remaining blocks are not in solution block_IDS (if not, disqualify solution)
                    other_blocks_IDs_in_sol = [(ID in sol_block_IDs) for ID in other_block_IDs]
                    if np.all(other_blocks_IDs_in_sol):

                        # Check if all of remaining blocks are in the correct state (if not, disqualify solution)
                        other_states_in_sol = [np.all(self.blocks[ID].state==sol_block_states[ID]) for ID in other_block_IDs]
                        if np.all(other_states_in_sol):

                            sol2add.append(sol_ID) # valid solution in excluded list

            if to_add:
                return sol2add
            else:
                rem_sol_IDs = list(self.current_solution_states.keys()) + sol2add  # lists of new validated indices
                rem_sol_IDs.sort()
                return rem_sol_IDs


        # Check when placing block --------------------------------
        else:
            """
            Invalid solutions must...
            - Not contain block ID in solution_block_IDs
            - Placement state must be equal to the state of that block_ID in solution
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


            if to_remove: return sol2remove # return which solutions to remove
            else: return rem_sol_IDs # return which solutions are valid


    ########################################################################
    ## Gamestate Updates ###################################################
    ########################################################################
    def update_block(self,block_ID,state):
        """
        Update the block state in the workspace, handle add/remove routing
        :param block_ID: ID of block to update
        :param state: new state of block
        """
        if state is None:  self.update_remove_block(block_ID)
        else: self.update_place_block(block_ID,state)

    def update_place_block(self,block_ID,state):
        assert block_ID not in self.placed_block_IDs, f'Block {block_ID} is already placed...'
        state = np.array(state)
        self.blocks[block_ID](state) # update block obj
        self.placed_block_IDs.append(block_ID)  # add to placed blocks
        self.placed_block_IDs.sort()  # Make sure lists are sorted
        self.unplaced_block_IDs.remove(block_ID) # remove from unplaced blokcs
        self.unfilled_mask += -1*self.blocks[block_ID].mask # update unfilled mask

        # Recalculate remaining valid solutions
        sol2remove = self.check_remaining_solutions(block_ID,state, to_remove=True)
        # print(f'removing solutiuons = {sol2remove}')
        for sol_ID in sol2remove:
            self.excluded_solution_states[sol_ID] = self.current_solution_states[sol_ID] # move to new dict
            del self.current_solution_states[sol_ID] # remove from old dict


    def update_remove_block(self,block_ID):
        assert block_ID not in self.unplaced_block_IDs, f'Block {block_ID} is already removed...'
        self.unplaced_block_IDs.append(block_ID)  # add to placed blocks
        self.unplaced_block_IDs.sort()  # Make sure lists are sorted
        self.placed_block_IDs.remove(block_ID)  # remove from unplaced blokcs

        self.unfilled_mask += self.blocks[block_ID].mask  # update unfilled mask
        self.blocks[block_ID](self.null_state)  # place block back in blockpool

        # Recalculate remaining valid solutions
        sol2add = self.check_remaining_solutions(block_ID, None, to_add=True)
        for sol_ID in sol2add:
            self.current_solution_states[sol_ID]  = self.excluded_solution_states[sol_ID] # move to new dict
            del self.excluded_solution_states[sol_ID]



    ########################################################################
    ## Utilities ###########################################################
    ########################################################################
    # def place_block(self,ws_mask,block):
    #     return ws_mask +

    def add_block(self,mask,block):
        return mask - block.mask
    def remove_block(self, mask, block):
        return mask + block.mask

    def get_block(self,block):
        return BlockDataClass(block['ID'], block['name'], block['state'])

    @property
    def blank_workspace(self):
        return self._ws_mask0.copy()

    @property
    def is_complete(self):
        return np.all(self.unfilled_mask==0)





# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    directory = '../SolutionSearch/Solutions/'
    # data_name = 'funnel__N137__D03-23-2024__T20-03-05.pkl'
    # data_name = 'funnel__N2167__D03-23-2024__T21-47-08.pkl'
    data_name = 'funnel__N263534__D03-24-2024__T04-55-17.pkl'
    structure, sols = load_solutions(data_name, dir=directory)

    agent = CoactiveAgent(structure, sols)

    human_actions = [(0, [3, 2, 1]), # incorrect
                     (1, [4, 4, 0]), # correct
                     (2, [3, 2, 3]), # correct
                     (7, [2, 6, 0]),  # incorrect
                     (4, [4, 5, 2]), # correct
                     ]

    for t in itertools.count():

        if t < len(human_actions):
            block_ID, state = human_actions[t]
            agent.update_block(block_ID=block_ID, state=state)
            print(f'\n[t={t}] HUMAN moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
        else:   print(f'\n Human out of moves')

        if agent.is_complete: print(f'\nCompleted structure!'); break

        block_ID, state = agent.pick()
        agent.update_block(block_ID=block_ID, state=state)
        print(f'[t={t}] AGENT moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
        # print(agent.unfilled_mask)

        if agent.is_complete: print(f'\nCompleted structure!'); break


    # DEPRICATED ------------------------------------------------------------

    # Seach for incompatible solutions to state 2 but complys with state 0
    # state ={}
    # state[0] = np.array([3, 2, 1])
    # state[2] = np.array([4, 3, 2])
    # state[3] = (1, 4, 3) # in solution 7
    # state[3] = (6, 5, 1)  # in solution 0
    # state[3] = (1, 6, 1)  # in solution 0
    # state[3] = (4, 2, 0)  # in solution 0
    # state[3] = (3,3,3)  # not compatable with block 2 but is with block 0 sol 7

    #
    # compatable_states = []
    # for sol_ID, sol_states in agent.current_solution_states.items():
    #     sol_block_IDs = list(sol_states.keys())
    #     if 0 in sol_block_IDs and not 2 in sol_block_IDs and 3 in sol_block_IDs:
    #         if np.all(state[0] == sol_states[0]) and not np.all(state[2] == sol_states[2]):
    #         if np.all(state[0] == sol_states[0]):
    #             print(sol_ID, sol_states[3])

    # print(f'Initial # solutions = {len(agent.current_solution_states)}')

    # Simulate placing blocks from solutions
    # agent.update_place_block(block_ID=0, state=state[0]) # in solution 7
    # print(f'Place block 0 ==> # solutions = {len(agent.current_solution_states)}')
    # agent.update_place_block(block_ID=2, state=state[2])  # in solution 7
    # print(f'Place block 2 ==> # solutions = {len(agent.current_solution_states)}')
    # agent.update_place_block(block_ID=3, state= state[3])
    # print(f'Place block 3 ==> # solutions = {len(agent.current_solution_states)}')

    # Check wich removal is best
    # rem_sol = agent.check_remaining_solutions(2,None, force_remove=True)
    # print(f'Remove block 2 --> Remaining solutions = {len(rem_sol)}')
    # rem_sol = agent.check_remaining_solutions(3, None, force_remove=True)
    # print(f'Remove block 3 --> Remaining solutions = {len(rem_sol)}')


    # Removes
    # agent.update_remove_block(3)
    # print(f'Remove block 3 --> Remaining solutions = {len(agent.current_solution_states)}')
    # agent.update_remove_block(2)
    # print(f'Remove block 2 --> Remaining solutions = {len(agent.current_solution_states)}')
    # agent.update_remove_block(0)
    # print(f'Remove block 0 --> Remaining solutions = {len(agent.current_solution_states)}')

    # block_ID,state = agent.pick()
    # print(f'Block {block_ID} with state {state} chosen')


if __name__ == "__main__":
    main()
