import warnings
from AgentModels.simulated_human import BoltzmannHuman
from AgentModels.coactive_agent import CoactiveAgent
import numpy as np
# import matplotlib.pyplot as plt
from SolutionSearch.utils.DataManagment import load_solutions
from SolutionSearch.utils.BlockAssets import BlockDataClass
import copy
import itertools



class DirectiveAgent(CoactiveAgent):
    """
    Inherits from CoactiveAgent and implements a directive agent that can be used to
    direct simulated human actions.
    """
    def __init__(self,structure,solutions):
        super().__init__(structure,solutions)
        self.human_model = BoltzmannHuman(self)
        self.human_model.rationality = 999
    def pick(self):
        if self.is_complete:
            warnings.warn('Agent tried to pick but assembly is finished...')
            return None, None


        # If there are still feasible solutions -----------------
        if len(self.current_solution_states) > 0:
            sol_IDs = list(self.current_solution_states.keys())
            sol_IDs = self.pick_block_imply_placement(sol_IDs)
            # sol_IDs = self.pick_solutions_with_max_reward(sol_IDs)
            # block_choices = self.pick_blocks_with_max_rem_sol(sol_IDs)
            # block_ID, state = self.pick_block_by_location(block_choices)
            # rem_sol = self.check_remaining_solutions(block_ID, state)
            # print(f'Block {block_ID} with state {state} chosen with n={len(rem_sol)} remaining solutions')
            return block_ID, state

        # Remove a block s.t. there are feasible solutions -----------------
        else:  # remove block
            num_sol_after_remove = {}
            for block_ID in self.placed_block_IDs:
                state = self.blocks[block_ID].state
                rem_sol = self.check_remaining_solutions(block_ID, None)
                num_sol_after_remove[block_ID] = len(rem_sol)
            # find the keys where num_sol_after_remove values == max_num_sol
            max_num_sol = max(num_sol_after_remove.values())
            remove_options = [block_ID for block_ID in num_sol_after_remove.keys() if
                              num_sol_after_remove[block_ID] == max_num_sol]
            block_ID = np.random.choice(remove_options)
            state = None
            # print(f'{num_sol_after_remove}')
            # print(f'Choosing to remove block {block_ID} with n={max_num_sol} remaining solutions')
            return block_ID, state

    def pick_block_imply_placement(self,sol_IDs):

        for block_ID in self.unplaced_block_IDs:
            for sol_ID in sol_IDs:
                if block_ID in self.current_solution_states[sol_ID]:
                    block_state = self.blocks[block_ID].state # possible block placed by AI
                    rem_unplaced_block_IDs = copy.copy(self.unplaced_block_IDs)
                    rem_unplaced_block_IDs.remove(block_ID) # all other remaining blocks except the one placed by AI

                    for

def main():
    directory = '../SolutionSearch/Solutions/'
    # data_name = 'funnel__N137__D03-23-2024__T20-03-05.pkl'
    data_name = 'funnel__N2167__D03-23-2024__T21-47-08.pkl'
    # data_name = 'funnel__N263534__D03-24-2024__T04-55-17.pkl'
    structure, sols = load_solutions(data_name, dir=directory)

    agent = DirectiveAgent(structure, sols)

    human_actions = [(0, [3, 2, 1]), # incorrect
                     (1, [4, 4, 0]), # correct
                     (2, [3, 2, 3]), # correct
                     (7, [2, 6, 0]),  # incorrect
                     (4, [4, 5, 2]), # correct
                     ]
    agent.pick()

    # # Simulate human and agent actions
    # for t in itertools.count():
    #
    #     if t < len(human_actions):
    #         block_ID, state = human_actions[t]
    #         agent.update_block(block_ID=block_ID, state=state)
    #         print(f'\n[t={t}] HUMAN moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
    #     else:   print(f'\n Human out of moves')
    #
    #     if agent.is_complete: print(f'\nCompleted structure!'); break
    #
    #     block_ID, state = agent.pick()
    #     agent.update_block(block_ID=block_ID, state=state)
    #     print(f'[t={t}] AGENT moving block {block_ID} to state {state} | NSOL = {len(agent.current_solution_states)}')
    #     # print(agent.unfilled_mask)
    #
    #     if agent.is_complete: print(f'\nCompleted structure!'); break

if __name__ == "__main__":
    main()