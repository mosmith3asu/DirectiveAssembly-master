import numpy as np
from SolutionSearch.utils.DataManagment import load_solutions
from AgentModels.coactive_agent import CoactiveAgent
import copy
import itertools
np.random.seed(0)
class BoltzmannHuman:
    def __init__(self,agent):
        """
        :param agent: AI agent to inherit necessary data from
        """
        self.enclosure_multiplier = 3

        self.rationality = 1
        self.structure = agent.structure
        self.blocks = agent.blocks
        self.solutions = agent.solutions
        self.unfilled_mask = agent.unfilled_mask
        self.all_solution_states = agent.all_solution_states
        self.all_solution_rewards = agent.all_solution_rewards
        self.n_blocks = agent.n_blocks

        self.current_solution_states = agent.current_solution_states  # current list of solutions states
        self.excluded_solution_states = agent.excluded_solution_states  # solutions that are no longer valid
        self.placed_block_IDs = agent.placed_block_IDs  # blocks currently paced in workspace
        self.unplaced_block_IDs = agent.unplaced_block_IDs  # blocks out of the workspace (blockpool)

        # self.current_solution_states = copy.deepcopy(agent.current_solution_states)  # current list of solutions states
        # self.excluded_solution_states = copy.deepcopy(agent.excluded_solution_states)  # solutions that are no longer valid
        # self.placed_block_IDs = copy.deepcopy(agent.placed_block_IDs)  # blocks currently paced in workspace
        # self.unplaced_block_IDs = copy.deepcopy(agent.unplaced_block_IDs)  # blocks out of the workspace (blockpool)

        # self.current_solution_states = self.all_solution_states  # current list of solutions states
        # self.excluded_solution_states = {}                              # solutions that are no longer valid
        # self.placed_block_IDs = []                                      # blocks currently paced in workspace
        # self.unplaced_block_IDs = [ID for ID in range(self.n_blocks)]   # blocks out of the workspace (blockpool)



    ########################################################################
    ## Action Selection ####################################################
    ########################################################################
    def pick(self):
        # Find block areas
        # block_areas = np.array([np.sum(self.blocks[block_ID].mask) for block_ID in self.unplaced_block_IDs])

        # Boltzmann distribution sampling
        # p_pick = np.exp(self.rationality * block_areas) / np.sum(np.exp(self.rationality * block_areas))
        # block_ID = np.random.choice(self.unplaced_block_IDs, p=p_pick)

        # Select valid state for that block
        selection_scores = {}
        selection_states = {}
        for block_ID in self.unplaced_block_IDs:
            best_state, best_score = self.pick_block_state_by_enclosure(block_ID)
            selection_scores[block_ID] = best_score
            selection_states[block_ID] = best_state

        # find the keys with the max value of selection_scores

        # !! BOLTZMAN SAMPLE ACCORDING TO SCORES !!
        # !! IF ENCLOSED, CHOOSE FROM THOSE !!
        # best_score = max(selection_scores.values())
        # best_block_IDs = [block_ID for block_ID in selection_scores.keys() if selection_scores[block_ID] == best_score]
        # block_ID = np.random.choice(best_block_IDs)
        # state = selection_states[block_ID]
        scores = np.array(list(selection_scores.values()))
        p_pick = np.exp(self.rationality * scores) / np.sum(np.exp(self.rationality * scores))
        block_ID = np.random.choice(list(selection_scores.keys()),p=p_pick)
        state = selection_states[block_ID]

        # Return placement
        return block_ID, state

    def pick_block_state_by_enclosure(self,block_ID):
        """
        Pick a state for a block based on the perimeter of the block
        # - Find the block placement that maximizes the perimeter between the block and the unfilled mask
        # - Given a state creates a new unfilled mask,
        - count the number of adjacent cells between the block and the filled mask
        :param block_ID: ID of block to pick state for
        :return: state: state of block
        """
        state0 = copy.copy(self.blocks[block_ID].state)

        # Get out of bounds mask
        out_of_bounds_mask = -1*np.copy(self.structure.mask) + 1

        # generate filled mask
        filled_mask = np.zeros_like(self.unfilled_mask)
        for placed_ID in self.placed_block_IDs:
            filled_mask += self.blocks[placed_ID].mask

        # get mask where all open tiles =0 otherwise 1
        compare_mask = filled_mask + out_of_bounds_mask

        # for each valid state for the block
        best_block_ID = None
        best_state = None
        best_score = 0 # count of adjacent cells between the block mask and the compare mask

        # Loop over all possible states for the block
        for this_state in self.get_legal_states(block_ID):
            self.blocks[block_ID](this_state) # update block obj
            block_mask = self.blocks[block_ID].mask # get the block mask
            # Count the number of adjacent cells between the block mask and the compare mask
            this_score,is_enclosed = self.enclosure_score(block_mask,compare_mask)
            if this_score>best_score:
                best_state = this_state
                best_score = this_score

        self.blocks[block_ID](state0) # reset block obj
        return best_state,best_score
    def get_legal_states(self,block_ID):
        state0 = copy.copy(self.blocks[block_ID].state)
        block = self.blocks[block_ID]
        legal_states = []
        for x in list(np.arange(1, 8)):
            for y in list(np.arange(1, 8)) :
                for r in range(4):
                    block((x,y,r)) # set block to new state
                    if np.all(self.unfilled_mask-block.mask>=0): # check if block is good place
                        legal_states.append((x,y,r))
        block(state0)  # reset block obj
        return legal_states
    def enclosure_score(self,block_mask,compare_mask):
        """
        Count the number of adjacent cells between the block mask and the compare mask
        :param block_mask: mask of block
        :param compare_mask: mask of unfilled cells
        :return: count: number of adjacent cells between the block mask and the compare mask
        """
        adjacent_mask = np.zeros_like(block_mask)
        tile_locs = np.array(np.where(block_mask==1)).T
        next_mask = compare_mask + block_mask
        # tile_is_enclosed = np.ones(block_mask)
        is_enclosed = True
        for itile, tile in enumerate(tile_locs):
            above = (tile[0]-1,tile[1])
            below = (tile[0]+1,tile[1])
            left = (tile[0], tile[1]-1)
            right = (tile[0],tile[1]+1)
            adjacent_count = np.arange(4)
            for neighbor in [above,below,left,right]:
                if compare_mask[neighbor]==1:
                    adjacent_mask[neighbor] = 1
                # check if tile is next to open tile (not enclosed)
                if next_mask[neighbor]==0:
                    is_enclosed = False


        n_neighbors = np.sum(adjacent_mask)
        if is_enclosed: n_neighbors = self.enclosure_multiplier*n_neighbors # bonus for enclosed blocks
        return n_neighbors,is_enclosed

    ########################################################################
    ## Gamestate Updates ###################################################
    ########################################################################
    def update_block(self,block_ID,state):
        """
        Update the block state in the workspace, handle add/remove routing
        :param block_ID: ID of block to update
        :param state: new state of block
        """
        # if state is None:  self.update_remove_block(block_ID)
        # else: self.update_place_block(block_ID,state)

        # ! DO NOTHING, ALREADY UPDATED BY agent.update_block(.) !
        pass

def main():
    directory = '../SolutionSearch/Solutions/'
    # data_name = 'funnel__N137__D03-23-2024__T20-03-05.pkl'
    data_name = 'funnel__N2167__D03-23-2024__T21-47-08.pkl'
    # data_name = 'funnel__N263534__D03-24-2024__T04-55-17.pkl'
    structure, sols = load_solutions(data_name, dir=directory)

    # Spawn agents
    agent = CoactiveAgent(structure, sols)
    human = BoltzmannHuman(agent)

    # # Check if updates correctly
    # states = {}
    # states[0] = np.array([3, 2, 1])
    # states[2] = np.array([4, 3, 2])
    # # states[3] = np.array([1, 4, 3]) # L_2x2
    # # states[4] = np.array([4, 2, 0]) # L_2x2 - enclosed by 0/2
    # for block_ID in states.keys():
    #     agent.update_block(block_ID=block_ID, state=states[block_ID])
    #
    # block_ID,state = human.pick()
    # agent.update_block(block_ID=block_ID, state=state)
    # print(f'human picked Block ID: {block_ID}, State: {state}')
    # # print(len(agent.placed_block_IDs))
    # # print(len(human.placed_block_IDs))
    # # print(agent.unfilled_mask)
    # # print(human.unfilled_mask)
    # # print(f'{agent.current_solution_states[7]}')
    # # print(agent.block_names)

    # Simulate human and agent actions
    for t in itertools.count():
        block_ID, state = human.pick()
        # human.update_block(block_ID=block_ID, state=state)
        agent.update_block(block_ID=block_ID, state=state)
        print(f'\n[t={t}] HUMAN moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
        # if t < len(human_actions):
        #     block_ID, state = human_actions[t]
        #     agent.update_block(block_ID=block_ID, state=state)
        #     print(f'\n[t={t}] HUMAN moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
        # else:   print(f'\n Human out of moves')

        if agent.is_complete: print(f'\nCompleted structure!'); break

        block_ID, state = agent.pick()
        # human.update_block(block_ID=block_ID, state=state)
        agent.update_block(block_ID=block_ID, state=state)
        print(f'[t={t}] AGENT moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
        # print(agent.unfilled_mask)

        if agent.is_complete: print(f'\nCompleted structure!'); break


if __name__ == '__main__':

    main()