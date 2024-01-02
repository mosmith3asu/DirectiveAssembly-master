import pickle
import warnings
from datetime import datetime
from itertools import product
from math import sqrt, ceil, floor
import scipy.ndimage
from Assembly import Assets
from Assembly.AssemblyUtils import *

class StructureSolver:
    def __init__(self, G, Bi, Blocks, SturctureName="Undefined"):
        ####################################################################
        # Settings #########################################################
        self.npad = 2               # padding around G to resolve small blocks in 3x3 container
        self.padval = -10           # value assigned to padding during check
        self.blk_val = 1            # base value representing blocks during check
        self.start_date = str(datetime.now().strftime("%m_%d_%Y"))  # time solver intiated
        self.file_name = "SS_"+SturctureName +"_"+ self.start_date + ".pkl" # name of saved file
        self.structure_name = SturctureName # Name of structure
        self.iter_report = 10000     # report in terminal every n iterations
        self.iter_save = 1e8        # save current StructureSolver obj every n iterations

        ####################################################################
        # Initialize #######################################################
        self.G = G                  # goal structure
        self.Bi = Bi                # list of block names (string) that is available to assembly
        self.nb = len(Bi)
        self.Blocks = Blocks        # block asset object from Assets.py
        self.Gpad = np.pad(-1 * self.G,
                           [(self.npad, self.npad), (self.npad, self.npad)],
                           mode='constant', constant_values=self.padval)
        self.Gpad_empty = 0*self.Gpad
        self.Gpad_sz = np.shape(self.Gpad) # size of padded goal space
        self.Solutions = []         # number of structures that complete G
        self.blki_placements = []   # all combinations of block in form [ib, [name,dx,dy,ri]]
        self.all_comb = []          # restructured blki_placements with index [ib, [name,dx,dy,ri]]=>i
        self.ncomb = 0              # number of possible block combinations of valid placements
        self.nsol = 0               # number of solutions found
        self.iprog = 0              # combinations complete (icomb)

        ####################################################################
        # Enable Valid WS Checks ###########################################
        ####################################################################

        self.invalid_spaces= {
            "I_1x1": True,
            "I_2x1": True,
            "I_3x1": True
        }
        if "I_2x1" in self.Bi: self.invalid_spaces["I_2x1"] = False
        if "I_3x1" in self.Bi: self.invalid_spaces["I_3x1"] = False

    def solve(self):
        #####################################################
        ## CHECK BLOCK EXACTLY COMPLETES STRUCTURE ##########
        print(">>Checking Feasability")
        if not (self.isFeasable(self.G, self.Bi, self.Blocks)):
            warnings.warn("AssemblyUtils>>solve_structure>>Bi cannont complete G")
            return None


        print(">>Finding Valid Combinations")

        #####################################################
        ## INITIALIZE SUBASSEMBLIES  ########################
        SA = []
        for blki, name in enumerate(self.Bi):
            SA.append(self.init_block2SA(name))

        #####################################################
        ## FORMULATE SUBASSEMBLY COMBINATION PLAN ###########
        all_SA_procedures = self.combine_SA_procedure(SA)
        n_rounds = len(all_SA_procedures)


        #####################################################
        ## ITERATE SUBASSEMBLY SOLUTIONS#####################
        for round, SA_procedure in enumerate(all_SA_procedures):
            print(f'\n\n\n######################################################################')
            print(f'########## Running Solver - ROUND {round + 1}/{n_rounds} ################################')
            print(f'######################################################################')
            print(f'\t\nSA_procedure={SA_procedure}')
            new_SA = []
            expected_nSA = len(SA_procedure)

            i_check = 0
            n_checks= self.calc_n_checks(SA, SA_procedure)
            print(f'N Checks = {n_checks}')

            for iproc,to_combine in enumerate(SA_procedure):
                for iSA in to_combine:
                    print(f'\t|SA{iSA}| = {len(SA[iSA])}')

            for to_combine in SA_procedure:
                # combined_SA=self.SA_Solver(SA, to_combine, i_check, n_checks,round,n_rounds)
                combined_SA = []
                SA_combine = self.access_list_indices(SA, to_combine)
                for icomb, comb in enumerate(product(*SA_combine)):
                    tmp_Placement, tmp_Block = ([], [])
                    for assem in comb:
                        tmp_Placement.append(assem["WS"])
                        tmp_Block += assem["Blocks"]
                    combined_Placements = sum(tmp_Placement)
                    combined_Blocks = tmp_Block
                    valid = self.check_WS(self.Gpad + combined_Placements, self.npad)
                    if valid:
                        combined_SA.append({"WS": combined_Placements, "Blocks": combined_Blocks})

                    i_check += 1
                    self.progress_update(i_check, n_checks, note=f'Round {round}/{n_rounds}')
                new_SA.append(combined_SA)

            # for iSA in range(len(new_SA)):
            #     print(f'\tnew SA{iSA}\tValid Assemblies={len(new_SA[iSA])}')

            SA = new_SA


        #####################################################
        ## CLOSE PROGRAM ####################################

        print(f'\n\n\n############################################################')
        print(f'########## CLOSE PROGRAM  ###########################################')
        print(f'#####################################################################')

        print(f'Solution Shape = {np.shape(SA)}')
        self.Solutions = SA[0]
        self.save()
        return self.Solutions

    def SA_Solver(self,SA,to_combine,i_check,n_checks,round,n_rounds):
        combined_SA = []
        SA_combine = self.access_list_indices(SA, to_combine)
        for icomb, comb in enumerate(product(*SA_combine)):
            tmp_Placement, tmp_Block = ([], [])
            for assem in comb:
                tmp_Placement.append(assem["WS"])
                tmp_Block += assem["Blocks"]
            combined_Placements = sum(tmp_Placement)
            combined_Blocks = tmp_Block
            valid = self.check_WS(self.Gpad + combined_Placements, self.npad)
            if valid:
                combined_SA.append({"WS": combined_Placements, "Blocks": combined_Blocks})


            i_check += 1
            self.progress_update(i_check, n_checks, note=f'Round {round}/{n_rounds}')


    def calc_n_checks(self,SA,SA_procedure):
        n_checks = 0
        for to_combine in SA_procedure:
            SA_combine = self.access_list_indices(SA, to_combine)
            n_combinations = 1
            for iSA in range(len(SA_combine)):
                n_valid_placements = len(SA_combine[iSA])
                n_combinations = n_combinations * n_valid_placements
            n_checks += n_combinations
        return n_checks
    def init_block2SA(self, name):
        blk = self.Blocks.get_rotations(name)
        SAi = []
        placements = []
        for ri, blk_rot in enumerate(blk):
            blk_sz = np.shape(blk_rot)
            for dx in range(self.Gpad_sz[1] - blk_sz[1]):
                for dy in range(self.Gpad_sz[0] - blk_sz[0]):
                    add_blk = np.pad(blk_rot * self.blk_val,
                                     [(dx, self.Gpad_sz[1] - blk_sz[1] - dx),
                                      (dy, self.Gpad_sz[0] - blk_sz[0] - dy)],
                                     mode='constant')
                    tmp_WS = self.Gpad + add_blk
                    valid_bounds = self.check_bounds(tmp_WS, self.npad, self.padval)
                    valid_place = self.check_WS(tmp_WS, self.npad)
                    if valid_bounds and valid_place:
                        block_state = [(name, dx, dy, ri)]
                        SAi.append({"WS": add_blk, "Blocks": block_state})

        print(name, "Valid Placements=", np.shape(SAi)[0])
        return SAi
    def combine_SA_procedure(self,SA, n=2, append_odd=False):
        print(f'\n### combine_SA_procedure() ###')
        all_SA_procedures = []
        nSA = len(SA)
        while nSA > 1:
            SA_procedure = []
            for i in range(floor(nSA / n)):
                start, end = (n * i, n * i + 1)
                if end + 2 == nSA and append_odd == True:  # adjust to n
                    igroup = (start, end, end + 1)
                    SA_procedure.append(igroup)
                elif end + 2 == nSA and append_odd == False:
                    igroup1 = (start, end)
                    igroup2 = tuple([end + 1])
                    SA_procedure.append(igroup1)
                    SA_procedure.append(igroup2)
                else:
                    igroup = (start, end)
                    SA_procedure.append(igroup)

            nSA = len(SA_procedure)
            print(f'nSA={nSA}\t Procedure={SA_procedure}')
            all_SA_procedures.append(SA_procedure)

        return all_SA_procedures
    def access_list_indices(self,lst, sub_indices, i_interest=0):

        if isinstance(sub_indices, int):
            return lst[sub_indices]
        else:
            # new_lst = []
            # for i in range(len(sub_indices)):
            #     new_lst.append(lst[i][i_interest])
            # return new_lst
            sub_indices = list(sub_indices)
            accessed_mapping = map(lst.__getitem__, sub_indices)
            return list(accessed_mapping)
    def isFeasable(self,G, Bi, Blocks):
        Bsum = 0
        for name in Bi: Bsum += np.sum(Blocks.Dict[name])
        if np.sum(G) != Bsum:
            warnings.warn("AssemblyUtils>>solve_structure>>Bi canont complete G")
            print("Gsum=", np.sum(G), "Bsum=", Bsum)
            return False
        return True
    def check_bounds(self, WS, npad, padval):
        # Check padded area to make sure block is placed in workspace
        rows, cols = np.shape(WS)
        Top = np.all(WS[0:npad, :] == padval)
        Bot = np.all(WS[rows - npad:rows, :] == padval)
        Left = np.all(WS[:, 0:npad] == padval)
        Right = np.all(WS[:, cols - npad:cols] == padval)
        valid = np.all([Top, Bot, Left, Right])
        return valid
    def check_WS(self, WS, npad):
        # check inside workspace to make sure block is in structure
        rows, cols = np.shape(WS)
        WS = WS[npad:rows - npad, npad:cols - npad]  # extract unpadded
        valid = np.all(WS <= 0)
        # Check if open spaces invalidate WS
        for region in self.extract_regions(WS):
            region_shape = np.shape(region)
            if self.invalid_spaces["I_1x1"]:
                if region_shape == (1, 1): return False
            if self.invalid_spaces["I_2x1"]:
                if region_shape == (2, 1) or region_shape == (1, 2): return False
            if self.invalid_spaces["I_3x1"]:
                if region_shape == (3, 1) or region_shape == (1, 3): return False
            if self.invalid_spaces["L_2x2"]:
                _, unique_count = np.unique(region, return_counts=True)
                has1missing = (np.size(unique_count) == 2 and min(unique_count) == 1)
                if region_shape == (2, 2) and has1missing: return False

        return valid
    def detect_region_shapes(self,WS):
        # Assign unique values over all similar regions
        regions = scipy.ndimage.label(WS)[0]
        labels = np.unique(regions)
        shapes = []
        for label in labels:
            region = regions
            del_r = np.all(regions != label, axis=1)
            region = np.delete(region, del_r, 0)
            del_c = np.all(regions != label, axis=0)
            region = np.delete(region,del_c,1)
            shapes.append(np.shape(region))
        return shapes



    def preview_solutions(self):
        print(f'\n\nOPENING PREVIEW PROGRAM...')
        preview_arrays = []
        for isol, sol in enumerate(self.Solutions):
            tmp_Blocks = sol["Blocks"]
            tmp_WS = self.Gpad
            for i_blk,block_state in enumerate(tmp_Blocks):
                name, dx, dy, ri=block_state
                tmp_WS = self.place_block(tmp_WS, name, dx, dy, ri, val=i_blk + 2)
            preview_arrays.append(tmp_WS)
        Theta = append_buffer(preview_arrays)
        preview_grid(Theta, Title='Assembly Permutations')
    def place_block(self, WS, name, dx, dy, ri, val=1):
        dx, dy, ri = [int(dx), int(dy), int(ri)]
        blk = val * self.Blocks.get_rotations(name, ri=ri)
        blk_sz = np.shape(blk)
        add_blk = np.pad(blk * self.blk_val,
                         [(dx, self.Gpad_sz[1] - blk_sz[1] - dx),
                          (dy, self.Gpad_sz[0] - blk_sz[0] - dy)],
                         mode='constant')
        new_WS = WS + add_blk
        return new_WS
    def progress_update(self, iter,total_iter,note="",nsig=1):
        self.iprog = iter
        if iter % self.iter_save == 0:
            self.save()
        if iter % self.iter_report == 0:
            percent_prog = round(iter / total_iter * 100, nsig)
            print(f'\t|PROG:{percent_prog}%\t'+note)  # Progress update
            #print('\t|PROG:',percent_prog, "%")  # Progress update
    def save(self):
        print("################### SAVED PROGRESS ##################")
        print("file name =", self.file_name)
        print("################### SAVED PROGRESS ##################")
        with open(self.file_name, 'wb') as SS_file:
            pickle.dump(self, SS_file)
    def resume(self):
        print("Resuming at iter = ", self.iprog)
        for icomb, comb in enumerate(self.all_comb[self.iprog:, :]):
            valid = self.check_comb(icomb, comb, self.Gpad)
            self.progress_update(icomb + self.iprog)




if __name__ == "__main__":

    # Structure Definitions -----------------------
    get_structure = 0 # Demo
    # get_structure = 1 # Filled1
    # get_structure = 2 # Spaceship
    # get_structure = 3 # Cross


    # Initialize Structure -----------------------
    AllStructures = Assets.Structures()
    G = AllStructures.Structure[get_structure]
    Bi = AllStructures.BlockSet[get_structure]
    Blocks = Assets.Blocks(BP=Bi)
    SS = StructureSolver(G, Bi, Blocks,SturctureName=AllStructures.names[0])
    #preview_grid(G)

    # Solve Structure -----------------------
    SS.solve()
    with open(SS.file_name, 'wb') as SS_file:
        pickle.dump(SS, SS_file)

    SS.preview_solutions()
