# Imports ######################
import numpy as np
import random
import numbers
import warnings
# Main #########################
def main():
    b=BlockDataClass(1,"L_2x2",[1,2,4],'r')
    print(b)
    print(b.data)
    new_state = [0,0,0]
    print(f'{b[:]}=>{b * 2} and {b*[1,3]}= {b*[1,3,5]}')
    print(f'{b(new_state)}')

    # print(type(np.array([1,1],dtype='int8')))
    # """ Main Function Call """
    # from Algorithm.Enviorment.PlottingUtils import simple_array_plot
    # blocks = BlockSets()
    # blocks.load_funnel1()
    # print(blocks.BlockPool)
    # print(blocks.color)
    # simple_array_plot(blocks.BlockPool,label_colors= blocks.color)

# Functions ####################
class BlockSets:
    def __init__(self):
        self.BlockAssetDict= {}
        self.BlockAssetDict["I_2x1"] = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.BlockAssetDict["I_3x1"] = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.BlockAssetDict["L_2x2"] = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.BlockAssetDict["L_3x2"] = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])
        self.BlockAssetDict["L_3x3"] = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
        self.BlockAssetDict["P_3x2"] = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]])
        self.BlockAssetDict["U_2x2"] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        self.BlockAssetDict["U_3x2"] = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]])
        self.BlockAssetDict["U_3x3"] = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]])
        self.BlockAssetDict["H_3x3"] = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]])
        self.BlockAssetDict["T_3x2"] = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        self.BlockAssetDict["T_3x3"] = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
        self.BlockAssetDict["Y_3x3"] = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
        self.BlockAssetDict["Z_3x2"] = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
        self.BlockAssetDict["Z_3x3"] = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
        self.BlockAssetDict["W_3x3"] = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        self._ColorAssets_ = ['orange','blue','black','pink','yellow','green']
        self.BP = np.zeros((20,20),'int8')
        self.BP_sz = np.shape(self.BP)
        self.names = None
        self.init_state = None
        self.bb_size = 3 # size of blocks bounding box

    def load_funnel1(self,init_states=None, colors=None):
        self.names=[]
        self.names += ["I_2x1" for _ in range(3)]
        self.names += ["I_3x1" for _ in range(2)]
        self.names += ["L_2x2" for _ in range(6)]
        self.names += ["L_3x2" for _ in range(1)]
        self.names += ["W_3x3" for _ in range(3)]
        self.names += ["T_3x3" for _ in range(1)]
        self.names += ["Y_3x3" for _ in range(1)]

        if init_states is None: self.init_state, self.BlockPool = self.auto_generate_blockpool(self.names,inc=4)
        #else: self.init_states,self.BlockPool = load_blockpool(self.Names,init_states)

        if colors is None:  self.color = [random.choice(self._ColorAssets_)for _ in self.names]
        else:  self.color = colors

    def auto_generate_blockpool(self,names, inc):
        ##############################################
        def check_x_bounds(place_state,dxy,sz):

            return place_state, dxy
        ##############################################
        BP = self.BP
        init_states = []
        place_state = np.array([0, 0])
        for ID, blk_name in enumerate(names):
            xrange = slice(place_state[0],place_state[0]+self.bb_size)
            yrange = slice(place_state[1],place_state[1]+self.bb_size)
            asset = self.BlockAssetDict[blk_name]
            BP[xrange,yrange] = (ID+1)*asset
            init_states += [place_state]
            dxy = np.array([0, inc])
            # Check X-Bounds -------------
            if place_state[1] + dxy[1] + 3 > self.BP_sz[1]:
                place_state[1] = 0
                dxy = np.array([inc, 0])
            # Check Y-Bounds -------------
                # @ADD CHECK
            place_state += dxy
        return init_states,BP




class BlockDataClass:
    AssetDict = {
        "I_2x1": np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype='int8'),
        "I_3x1": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype='int8'),
        "L_2x2": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype='int8'),
        "L_3x2": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]], dtype='int8'),
        "L_3x3": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype='int8'),
        "P_3x2": np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype='int8'),
        "U_2x2": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype='int8'),
        "U_3x2": np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype='int8'),
        "U_3x3": np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype='int8'),
        "H_3x3": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype='int8'),
        "T_3x2": np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype='int8'),
        "T_3x3": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype='int8'),
        "Y_3x3": np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype='int8'),
        "Z_3x2": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype='int8'),
        "Z_3x3": np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype='int8'),
        "W_3x3": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype='int8'),
    }
    PolyDict = {
        "I_2x1": [[1, 1], [1, 3], [2, 3], [2, 1]],
        "I_3x1": [[1, 0], [1, 3], [2, 3], [2, 0]],
        "L_2x2": [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]],
        "L_3x2": [[1, 0], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]],
        "L_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [3, 1], [3, 0]],
        "P_3x2": [[0, 0], [0, 3], [2, 3], [2, 1], [1, 1], [1, 0]],
        "U_2x2": [[0, 1], [0, 3], [2, 3], [2, 1]],
        "U_3x2": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 1]],
        "U_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [2, 1], [2, 3], [3, 3], [3, 0]],
        "H_3x3": [[0, 0], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]],
        "T_3x2": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2]],
        "T_3x3": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 0], [1, 0], [1, 2]],
        "Y_3x3": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 1], [2, 1], [2, 0], [1, 0], [1, 1]],
        "Z_3x2": [[0, 2], [0, 3], [2, 3], [2, 2], [3, 2], [3, 1], [1, 1], [1, 2]],
        "Z_3x3": [[0, 2], [0, 3], [2, 3], [2, 1], [3, 1], [3, 0], [1, 0], [1, 2]],
        "W_3x3": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1], [3, 0], [1, 0], [1, 1]]
    }
    def __init__(self, ID, name, state, color, world_sz=np.array([20,40])):
        self.__dict__['ID'] =ID
        self.__dict__['name']=name
        self.__dict__['dtype'] = 'int8'
        self.__dict__['state']=np.array(state,dtype=self.dtype)
        self.__dict__['color']=color
        self.__dict__['world_sz'] = world_sz

    def add(self,other):
        state = self.state
        if isinstance(other, numbers.Number):state[0:2] += other
        elif isinstance(other,list) and len(other)==2:                state[0:2] += np.array(other,dtype=self.dtype)
        elif isinstance(other, list) and len(other) == 3:           state += np.array(other,dtype=self.dtype)
        elif isinstance(other,np.ndarray) and np.size(other)==2:    state[0:2] += other
        elif isinstance(other, np.ndarray) and np.size(other) == 3: state += other
        else: state[0:2] += other.state[0:2]
        return state
    def sub(self,other):
        state= self.state
        if isinstance(other, numbers.Number): state[0:2] -= other
        elif isinstance(other, list) and len(other) == 2: state[0:2] -= np.array(other, dtype=self.dtype)
        elif isinstance(other, list) and len(other) == 3: state -= np.array(other, dtype=self.dtype)
        elif isinstance(other, np.ndarray) and np.size(other) == 2: state[0:2] -= other
        elif isinstance(other, np.ndarray) and np.size(other) == 3: state -= other
        else: state[0:2] -= other.state[0:2]
        return state
    def mult(self,other):
        state = np.array(self.state)
        if isinstance(other, numbers.Number):state[0:2] = state[0:2]*other
        elif isinstance(other, list): state[0:2] = np.multiply(state[0:2],np.array(other[0:2],dtype=self.dtype))
        elif isinstance(other, np.ndarray): state[0:2] = np.multiply(state[0:2],other[0:1])
        else: state[0:2] = np.multiply(state[0:2],other.state[0:2])
        return state
    def div(self,other):
        state = self.state
        if isinstance(other, numbers.Number): state[0:2] = state[0:2]/other
        elif isinstance(other, list): state[0:2] = np.divide(state[0:2],np.array(other[0:2],dtype=self.dtype))
        elif isinstance(other, np.ndarray): state[0:2] = np.divide(state[0:2],other[0:1])
        else: state[0:2] = np.divide(state[0:2],other.state[0:2])
        return state


    def __eq__(self, other): return np.all(self.state==other.state)
    def __add__(self, other): return self.add(other)
    def __iadd__(self, other): return self.add(other)
    def __sub__(self, other): return self.sub(other)
    def __isub__(self, other): return self.sub(other)
    def __mul__(self, other): return self.mult(other)
    def __imul__(self, other): return self.mult(other)
    def __idiv__(self, other): return self.div(other)
    def __truediv__(self, other): return self.div(other)
    def __setitem__(self, key, value): self.state[key] = value # call: self[key]
    def __getitem__(self, key): return self.state[key] # call: val = self[key]
    def __repr__(self): return f'({self.ID}){self.name}={self.state}'
    def __str__(self): return f'({self.ID}){self.name}={self.state}'
    def __call__(self, state):
        self.state = np.array(state, dtype=self.dtype)
        return self.state
    def __setattr__(self, name, value): # call: self.name = val
        if name=='state': self.__dict__[name] = np.array(value,dtype=self.dtype)
        else: self.__dict__[name] = value
    def __iter__(self):
        for xyr in self.state: yield xyr

    @property
    def data(self):return {'ID':self.ID,'name':self.name,'color':self.color,'state':self.state}
    @property
    def layer(self):
        """ Make the asset the same size as world """
        # Unpack Vars
        x, y, r = self.state
        asset =np.rot90(BlockDataClass.AssetDict[self.name], k=r)
        world_h, world_w = self.world_sz
        asset_h, asset_w = np.shape(asset)
        # Calculate pad dimensions
        npad_h, npad_w = world_h - asset_h, world_w - asset_w
        pad_left, pad_right,pad_bot, pad_top = x, npad_w - x,y, npad_h - y
        # Pad and Return
        padded_asset = np.pad(asset, [(pad_bot, pad_top), (pad_left, pad_right)], mode='constant')
        return padded_asset

    @property
    def mask(self):
        return self.layer

    @property
    def sum(self):
        return np.sum(BlockDataClass.AssetDict[self.name])


class BlockObj:
    AssetDict = {
        "I_2x1": np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]],dtype='int8'),
        "I_3x1": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]],dtype='int8'),
        "L_2x2": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]],dtype='int8'),
        "L_3x2": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]],dtype='int8'),
        "L_3x3": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]],dtype='int8'),
        "P_3x2": np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]],dtype='int8'),
        "U_2x2": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]],dtype='int8'),
        "U_3x2": np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]],dtype='int8'),
        "U_3x3": np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]],dtype='int8'),
        "H_3x3": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]],dtype='int8'),
        "T_3x2": np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]],dtype='int8'),
        "T_3x3": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]],dtype='int8'),
        "Y_3x3": np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]],dtype='int8'),
        "Z_3x2": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]],dtype='int8'),
        "Z_3x3": np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]],dtype='int8'),
        "W_3x3": np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]],dtype='int8'),
    }
    PolyDict = {
        "I_2x1": [[1, 1], [1, 3], [2, 3], [2, 1]],
        "I_3x1": [[1, 0], [1, 3], [2, 3], [2, 0]],
        "L_2x2": [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]],
        "L_3x2": [[1, 0], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]],
        "L_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [3, 1], [3, 0]],
        "P_3x2": [[0, 0], [0, 3], [2, 3], [2, 1], [1, 1], [1, 0]],
        "U_2x2": [[0, 1], [0, 3], [2, 3], [2, 1]],
        "U_3x2": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 1]],
        "U_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [2, 1], [2, 3], [3, 3], [3, 0]],
        "H_3x3": [[0, 0], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]],
        "T_3x2": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2]],
        "T_3x3": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 0], [1, 0], [1, 2]],
        "Y_3x3": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 1], [2, 1], [2, 0], [1, 0], [1, 1]],
        "Z_3x2": [[0, 2], [0, 3], [2, 3], [2, 2], [3, 2], [3, 1], [1, 1], [1, 2]],
        "Z_3x3": [[0, 2], [0, 3], [2, 3], [2, 1], [3, 1], [3, 0], [1, 0], [1, 2]],
        "W_3x3": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1], [3, 0], [1, 0], [1, 1]]
    }

    def __init__(self, ID, name, state, World_sz=(12, 23), no_fill_val=0):
        """ Object for each block in assembly enviorment
         :param self.ID: unique ID for this block
         :param self.name: name of block
         :param self.state: state of block with ABSOLUTE ROTATION
         :param self._asset_: base asset (CONST)
         :param self.current_asset: current rotation
         :param self.layer: array w. sz of world with -ID values for removal and +ID for addition of new block location
        """

        # Private and Static Variables --------------------
        self.ID = ID
        self.name = name
        self._asset_ = BlockObj.AssetDict[name]
        self._poly_ = BlockObj.PolyDict[name]
        self._ri_ = [0, 1, 2, 3]
        self._world_sz_ = World_sz

        # Public and Changing Parameters ---------------
        self.state = state
        self.current_asset = self._asset_
        self.current_poly = self._poly_
        self.layer = self.pad_asset(self._asset_)


    def move(self, action, world):
        """ Switch all values of old block layer to negative and add positive block ID in new state location
         :param world: State of world with all blocks placed
         :param action: action=[dx,dy,dr,gripped]
         :return new_world: World with this blocks asset moved
        """
        self.state = self.update_state(action)
        self.current_asset = self.get_asset_rotation(self.state[2])  # Get proper orientation of asset
        new_world = world + self.update_layer()  # add layer to remove old and add new block with ID to world
        return new_world

    def update_state(self, action):
        state = self.state[0:2] + action[0:2]  # update current state [x,y,r]
        state[2] = self.ri[state[2]]  # change to absolution oriantation of asset
        return state

    def update_layer(self):
        """ Layer that removes old state and adds new state when adding to world"""
        self.layer[self.layer == self.ID] = -self.ID  # flip old state
        self.layer += self.pad_asset(self.current_asset)  # get array with -ID for old state and +ID for new state
        return self.layer

    def pad_asset(self, asset):
        """ Make the asset the same size as world """
        # Unpack Vars
        x, y, r = self.state
        world_h, world_w = self.world_sz
        asset_h, asset_w = np.shape(asset)
        # Calculate pad dimensions
        npad_h, npad_w = world_h - asset_h, world_w - asset_w
        pad_left, pad_right = x, npad_w - x
        pad_bot, pad_top = y, npad_h - y
        # Pad and Return
        padded_asset = np.pad(asset, [(pad_bot, pad_top), (pad_left, pad_right)], mode='constant')
        return padded_asset

    def get_asset_rotation(self, irot):
        rotated_asset = np.rot90(self._asset_, k=irot)
        return rotated_asset

    def check_array(self, val):
        if isinstance(val, (np.ndarray, np.generic)):
            return val
        if isinstance(val, 'list'):
            return np.array(val)
        else:
            warnings.warn(f'In <BlockObj.check_array> unknown type {type(val)}')
            return val


# Run ##########################
if __name__ == '__main__':
    main()
