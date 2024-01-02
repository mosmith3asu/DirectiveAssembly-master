import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
from datetime import datetime
import time
from AgentModels.Agents import AgentsOBJ
from Assembly import StructureAssets,BlockAssets
plt.ion()

def main():
    Vi = ["Z_3x2", "U_2x2", "U_3x2", "L_2x2", "L_3x2", "W_3x3", 'T_3x3']
    BlockSet = BlockAssets.BlockSets()
    BlockSet.load_funnel1()
    agent_i = AgentsOBJ(1, DMM=None, state=[3, 3, 0, 0])
    agent_j = AgentsOBJ(2, DMM=None, state=[7, 7, 0, 0])
    env = AssemblyEnviorment(BlockSet,agent_i,agent_j)



    for _ in range(100):
        #env.World = np.random.randint(low=1, high=5,size=[10,21])
        env.agent_i.state = np.array(env.agent_i.state) + np.array([1,1,0,0])
        env.agent_j.state = np.array(env.agent_j.state) + np.array([0, -1, 0, 0])
        #print(env.agent_j.state)
        env.update_clock()
        env.draw()
        time.sleep(0.1)
    print(f'finished {__name__}...')



class AssemblyEnviorment:
    def __init__(self,BlockSet, agent_i,agent_j, nbuffer = 1):
        self.agent_i = agent_i
        self.agent_j = agent_j
        # WORLD STATE ----------------------------------------------------------------
        # self.Goal = Assembly.G
        # self.BlockSet = Assembly.Bi
        # self.BlockPool = Assembly.BP
        self.BlockPool = BlockSet.BP
        self.BlockColors = BlockSet.color
        self.WorkSpace = np.zeros(np.shape(self.BlockPool))


        cbuffer = -1*np.ones([np.shape(self.BlockPool)[0],nbuffer])
        self.World = np.concatenate((self.BlockPool,cbuffer,self.WorkSpace),axis = 1)
        self.World_sz = np.shape(self.World)
        self.clock = {"start": datetime.now(),
                      "duration": datetime.now(),
                      "t_last_draw": datetime.now(),
                      "t_since_draw": datetime.now()}
        # PLOTTING ----------------------------------------------------------------
        self.draw_fps = 0.1
        self.figure, self.ax = plt.subplots()
        self.tile_colors = ['gray', 'white'] + self.BlockColors
        self.tile_bounds = list(range(-1,np.size(self.tile_colors)- 1))
        self.grid_color = 'k'
        self.init_render("Collaborative Assembly")

    def init_render(self,title):
        plt.title(title, fontsize=20)
        # draw render
        cmap = colors.ListedColormap(self.tile_colors)
        norm = colors.BoundaryNorm(self.tile_bounds, cmap.N)
        self.World_img = self.ax.imshow(self.World, cmap=cmap, norm=norm)
        self.WS_border = Rectangle((self.World_sz[1]-15.5,4.5), 10, 10,
                                   linewidth=5, edgecolor='k',fill=False,linestyle ='--')

        self.WS_border_img = self.ax.add_patch(self.WS_border )
        self.Agent_i_img= self.ax.add_patch(self.agent_i.icon)
        self.Agent_j_img = self.ax.add_patch(self.agent_j.icon)

        # draw gridlines
        self.ax.grid(which='major', axis='both', linestyle='-', color=self.grid_color, linewidth=2)
        self.ax.set_xticks(np.arange(-.5, self.World_sz[1], 1))
        self.ax.set_yticks(np.arange(-.5, self.World_sz[0], 1))

    def draw(self):
        # create discrete colormap
        if self.clock["t_since_draw"] > self.draw_fps:
            self.World_img.set_data(self.World)
            try:
                self.Agent_i_img.set_xy((self.agent_i.state[0],self.agent_i.state[1])) # for rect
                self.Agent_j_img.center = self.agent_j.state[0], self.agent_j.state[1] # for circle
            except:
                self.Agent_j_img.set_xy((self.agent_j.state[0], self.agent_j.state[1])) # for rect
                self.Agent_i_img.center = self.agent_i.state[0], self.agent_i.state[1] # for circle

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.clock["t_last_draw"]=datetime.now()
            self.update_clock()

    def update_clock(self):
        duration = datetime.now()-self.clock["start"]
        self.clock["duration"] = duration.total_seconds()
        duration = datetime.now()-self.clock["t_last_draw"]
        self.clock["t_since_draw"] = duration.total_seconds()
        print(f'Duration: {self.clock["duration"]}\t Since LSt Draw: {self.clock["t_since_draw"]} ')


if __name__ == "__main__":
    main()