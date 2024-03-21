import numpy as np
import warnings
import warnings

def main():
    Blks = Blocks()
    rots = Blks.get_rotations("L_2x2")
    # for r in Blks.Rot["L_2x2"]: print(r)

    # Check if array matches block definition
    check_name ="L_2x2"
    check_arr =np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    check = Blks.checkif(name=check_name,arr=check_arr)
    print(check)




class Structures:
    def __init__(self):
        self.BlockSet = []
        self.Structure = []
        self.names = []

        ##############################################################################
        # Demo ####################################################################
        self.names.append("Demo")
        self.BlockSet.append(["L_2x2", "L_2x2", "I_2x1", "I_3x1", "U_2x2"])
        self.Structure.append(np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        ##############################################################################
        # Filled1 ####################################################################
        self.names.append("Filled1")
        self.BlockSet.append(["I_3x1", "I_3x1", "I_3x1", "I_3x1", "L_2x2", "L_2x2", "L_3x2", "L_3x2", "L_3x2", "U_2x2", "U_2x2"])
        self.Structure.append(np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]))
        # # # # self.BlockSet.append(["I_2x1", "I_2x1", #"I_2x1","I_3x1", "I_3x1","L_2x2", "L_2x2", "L_3x2", "L_3x2", "U_2x2","U_2x2","U_2x2"])
        # # # # self.BlockSet.append(["I_2x1", "I_3x1", "I_3x1", "I_3x1","L_2x2","L_3x2", "L_3x2","L_3x2","U_2x2","U_2x2","U_2x2"])

        ##############################################################################
        # Spaceship ####################################################################
        self.names.append("Spaceship")
        self.BlockSet.append(["I_3x1", "I_2x1", "U_2x2", "U_2x2", "Z_3x2", "Z_3x2", "U_3x2"])
        self.Structure.append(np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]))

        ##############################################################################
        # Cross ####################################################################
        self.names.append("Cross")
        # self.BlockSet.append(["I_3x1","I_3x1","L_2x2","L_2x2","L_2x2","L_3x2", "U_2x2","U_3x2","T_3x2"])# intractable
        self.BlockSet.append(["L_3x2","L_3x2","L_3x2","P_3x2","Z_3x3","U_3x2","T_3x3"])
        self.Structure.append(np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        ##############################################################################
        # Oval ####################################################################
        self.names.append("Oval")
        # self.BlockSet.append(["I_3x1","I_3x1","L_2x2","L_2x2","L_2x2","L_3x2", "U_2x2","U_3x2","T_3x2"])# intractable
        self.BlockSet.append(["T_3x2","W_3x3","W_3x3","L_3x2","L_3x2","P_3x2","Z_3x2","U_3x2"])
        self.Structure.append(np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        ##############################################################################
        # Coin #######################################################################
        self.names.append("Coin")
        #self.BlockSet.append(['L_2x2',"L_3x2","L_3x2",'I_3x1',"P_3x2","U_3x2","Z_3x2","Z_3x2"]) # 16 solutions
        #self.BlockSet.append( ['L_2x2','L_3x2','L_2x2','L_3x2', 'L_2x2',"Z_3x2", "Z_3x2",'U_2x2', 'L_2x2']) # 1536 solutions
        self.BlockSet.append([ "Z_3x2","Z_3x2","T_3x2",'L_2x2', 'L_2x2', 'L_2x2', 'L_3x2','L_3x2',"I_3x1" ])
        self.Structure.append(np.array(
            [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,1,1,0,0,1,1,1,0],
            [0,1,1,1,0,0,1,1,1,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]))

        ##############################################################################
        # Scatter ####################################################################
        self.names.append("Scatter")
        self.BlockSet.append(["T_3x2", "T_3x2", "T_3x2","P_3x2","P_3x2","L_3x2", "L_3x2", "Z_3x2",  "U_3x2"])
        self.Structure.append(np.array(
            [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,1,1,0,1,0,0,0,1,0],
            [0,1,1,1,1,0,0,1,1,0],
            [0,1,0,1,1,1,0,1,1,0],
            [0,0,0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1,1,0],
            [0,1,1,1,1,0,1,0,1,0],
            [0,0,0,1,1,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]))

        ##############################################################################
        # FUNNEL #####################################################################
        self.names.append("Funnel")
        # self.BlockSet.append(["Z_3x2","U_2x2","U_3x2", "L_2x2","L_3x2","W_3x3",'T_3x3']) # 7 solutions
        # self.BlockSet.append(["Z_3x2", 'Z_3x3', "W_3x3", "T_3x2","I_3x1","L_2x2","L_2x2","L_2x2"]) # 432 solutions
        self.BlockSet.append(["T_3x3","T_3x2","L_3x2", 'L_3x2','L_3x2', "L_2x2", 'L_2x2', 'L_2x2'])
        self.Structure.append(np.array(
            [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,0,0,0],
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]))


class Blocks:
    def __init__(self, BP=None):
        self.Dict = {}
        self.Dict["I_1x1"] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.Dict["I_2x1"] = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.Dict["I_3x1"] = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.Dict["L_2x2"] = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.Dict["L_3x2"] = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])
        self.Dict["L_3x3"] = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
        self.Dict["P_3x2"] = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]])
        self.Dict["U_2x2"] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        self.Dict["U_3x2"] = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]])
        self.Dict["U_3x3"] = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]])
        self.Dict["H_3x3"] = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]])
        self.Dict["T_3x2"] = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        self.Dict["T_3x3"] = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
        self.Dict["Y_3x3"] = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
        self.Dict["Z_3x2"] = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.Dict["Z_3x3"] = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        self.Dict["W_3x3"] = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])

        self.Rot = self.init_rotations(self.Dict)
        self.BP = BP
        if self.BP == None: warnings.warn("Please set BP in >>Assets.Blocks")

        self.Shapes = []
        self.Names = []
        self.ID = []
        for i, key in enumerate(self.Dict):
            self.Shapes.append(self.Dict[key])
            self.Names.append(key)
            self.ID.append(i)

    def checkif(self,name,arr):
        checks = [np.all(arr == block_def) for block_def in self.Rot[name]]
        is_in_list = np.any(checks)
        return is_in_list

    def trim_except(self,arr,except_val):
        region = arr
        del_r = np.all(arr != except_val, axis=1)
        region = np.delete(region, del_r, 0)
        del_c = np.all(arr != except_val, axis=0)
        region = np.delete(region, del_c, 1)
        return region


    def isUnique(self,blk,blk_list):
        tmp_blk = self.remove_buffer(blk)
        isExist = []
        for eblk in blk_list:
            eblk = self.remove_buffer(eblk)
            isExist.append(np.all(tmp_blk==eblk) and np.all(np.shape(tmp_blk)==np.shape(eblk)))
        if np.any(isExist): return False
        else: return True

    def remove_buffer(self,BLK):
        blk = np.copy(BLK)
        ikeep_row = np.any(blk != 0, axis=1)
        ikeep_col = np.any(blk != 0, axis=0)
        new_blk = blk[ikeep_row,:]
        new_blk = new_blk[:, ikeep_col]
        return new_blk


    def init_rotations(self,Dict):
        Rot = {}
        for name,block in Dict.items():
            Rot[name] = self.get_rotations(name)
        return Rot



    def get_rotations(self,name,ri=range(0,4)):
        BLK = self.Dict[name]
        if isinstance(ri, range):
            blks = []
            for irot in ri:
                rot = np.rot90(BLK, k=irot)
                if self.isUnique(rot, blks):
                    blks.append(rot)
        else:
            blks = np.rot90(BLK, k=ri)

        return blks



if __name__ == "__main__":
    main()

