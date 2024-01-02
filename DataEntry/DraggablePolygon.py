import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import floor
from random import randint as randi
from matplotlib.patches import Rectangle
import pandas


def main():
    fig = plt.figure()
    ax = fig.add_subplot()
    World_sz=[40,20]
    ax.set_xticks(np.arange(1, World_sz[0]))
    ax.set_yticks(np.arange(1, World_sz[1]))

    WS_border = Rectangle((25, 5), 10, 10,
                               linewidth=5, edgecolor='k', fill=False, linestyle='--')
    WS_border_img = ax.add_patch(WS_border)

    block_names = ["I_2x1","I_3x1","L_2x2","L_3x2","L_3x3" ,"P_3x2" ,"U_2x2" ,"U_3x2","U_3x3","H_3x3","T_3x2","T_3x3","Y_3x3","Z_3x2","Z_3x3", "W_3x3" ]
    offset = np.array([0,0])

    for ID,name in enumerate(block_names):
        dp = DraggablePolygon(ax, ID, name, c_offset=offset)
        dp.connect()
        offset += np.array([0, 3]) # move up
        if offset[1]>20-3: offset+= np.array([3,-offset[1]])

    plt.xlim([0, World_sz[0]])
    plt.ylim([0, World_sz[1]])

    plt.grid(True, markevery=1)
    plt.show()



#https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

class DraggablePolygon:
    all_objects = []
    # history = []
    pandas.DataFrame
    def __init__(self,ax,ID,name,color=None,c_offset= np.array([0,0])):
        """ Centered on [1.5,1.5] by default and
        parram: center: offset of block relative to [1.5,1.5]
        """
        self.lock = False
        self.press = None
        self._poly_defs_ = {
            "I_2x1": [[1, 1], [1, 3], [2, 3], [2, 1]],
            "I_3x1": [[1, 0], [1, 3], [2, 3], [2, 0]],
            "L_2x2": [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]],
            "L_3x2": [[1, 0], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]],
            "L_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [3, 1], [3, 0]],
            "P_3x2": [[0,0],[0,3],[2,3],[2,1],[1,1],[1,0]],
            "U_2x2": [[0, 1], [0, 3], [2, 3], [2, 1]],
            "U_3x2": [[0, 1], [0, 3], [1, 3],[1, 2],[2, 2], [2, 3],[3, 3],[3, 1]],
            "U_3x3": [[0, 0], [0, 3], [1, 3], [1, 1], [2, 1], [2, 3], [3, 3], [3, 0]],
            "H_3x3": [[0, 0], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]],
            "T_3x2": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 1],[1, 1],[1, 2]],
            "T_3x3": [[0, 2], [0, 3], [3, 3], [3, 2], [2, 2], [2, 0], [1, 0], [1, 2]],
            "Y_3x3": [[0, 1], [0, 3], [1, 3], [1, 2], [2, 2], [2, 3], [3, 3], [3, 1], [2, 1], [2, 0], [1, 0], [1, 1]],
            "Z_3x2": [[0, 2],[0, 3],[2, 3],[2, 2],[3, 2],[3, 1],[1, 1],[1, 2]],
            "Z_3x3": [[0, 2], [0, 3], [2, 3], [2, 1], [3, 1], [3, 0], [1, 0], [1, 2]],
            "W_3x3": [[0, 1], [0, 3], [1, 3], [1,2 ], [2, 2], [2, 1], [3, 1], [3, 0], [1, 0], [1, 1]]
        }

        # L_2x2 ----------------------------------------
        self.ID = ID
        self._is_master_ = (ID==1)
        self.name = name
        self.center = np.array([1.5, 1.5]) + c_offset
        self.geometry = self._poly_defs_[name]
        for i, xy in enumerate(self.geometry ):
            self.geometry[i][1] += c_offset[1]
            self.geometry[i][0] += c_offset[0]
        cx,cy = self.center
        self.bounding_box = [[cx-1.5,cy-1.5],[cx+1.5,cy-1.5],[cx+1.5,cy+1.5],[cx-1.5,cy+1.5]]
        self.all_colors = ['green','blue',"pink",'orange','black','yellow']
        self.nc = len(self.all_colors)
        if color is None: self.color = self.all_colors[randi(0,self.nc-1)]
        else: self.color = color


        # Draw Asset ----------------------------------
        self.CP = plt.scatter(self.center[0], self.center[1],cmap='k',alpha=0)
        self.poly_asset = plt.Polygon(self.geometry, closed=True, linewidth=2, facecolor=self.color , edgecolor='k' )
        self.poly_bb = plt.Polygon(self.bounding_box, closed=True, fill=False, linestyle='--', linewidth=3,
                                   facecolor=self.color, edgecolor=self.color , alpha=0)

        ax.add_patch(self.poly_asset)
        ax.add_patch(self.poly_bb)

        # Initi other vars ----------------------------
        self.newGeometry = []
        self.newCenter = []
        self.newBB = []
        self.ri = 0
        self.last_ri=self.ri
        self.last_center=self.center
        DraggablePolygon.all_objects.append(self)


    def connect(self):
        'connect to all the events we need'
        print(f'connecting... [{self.name}]')
        self.cidpress = self.poly_asset.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.poly_asset.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.poly_asset.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.keypress = self.poly_asset.figure.canvas.mpl_connect('key_press_event', self.on_key)


    def on_key(self,event):
        key, x0, y0 = event.key, event.xdata, event.ydata
        if self._is_master_:
            if key == 'c':
                print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!! history cleared !!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n')
                DraggablePolygon.history = DraggablePolygon.history[-1]
            elif key == 'h': print(DraggablePolygon.history)
            elif key == 'w': print(f'\n\n #### STATE REPORT ########\n{self.get_world()}')


        # Only why object is selected --------------------------
        if event.inaxes != self.poly_asset.axes: return
        if not self.lock: return
        if key == 'right': self.geometry = self.Rotate2D(self.geometry,self.center,dir=1)
        elif key == 'left': self.geometry = self.Rotate2D(self.geometry, self.center, dir=-1)
        elif key == 'up': self.color = self.all_colors[min(self.nc-1,self.all_colors.index(self.color)+1)]
        elif key == 'down': self.color = self.all_colors[max(0, self.all_colors.index(self.color) -1)]
        elif key == 'enter' or key==' ':
            # Update canvas ---------------------------------------------
            self.center, self.geometry, self.bounding_box = self.update_loc(event)
            self.lock = False
            self.redraw()

            # Record Action and State ---------------------------------------------
            world = self.get_world()
            action = self.get_action()
            DraggablePolygon.history.append({"action": action,"state": world.replace('\n','')})
            print(DraggablePolygon.history)
            self.last_center = self.center
            self.last_ri = self.ri
            return

        self.redraw(new=True)

    def get_action(self):
        dx, dy, dr = self.center[0] - self.last_center[0], self.center[1] - self.last_center[1], self.ri - self.last_ri
        A = {"ID": self.ID, "name": self.name, "color": self.color, "dxyr": np.array([dx, dy, dr]),"time":0.0}
        return A
    def get_world(self):
        world = '['
        for ib, poly_obj in enumerate(DraggablePolygon.all_objects):
            world += '{"name": ' + f'{poly_obj.name},' + \
                     '"state": ' + f'np.array([{poly_obj.center[0]},{poly_obj.center[1]},{poly_obj.ri}]),' + \
                     '"color": ' + f'"{poly_obj.color}"' + '},\n'
        world += ']'
        return world



    def Rotate2D(self,pts, cnt, dir = 1):
        self.ri+=dir
        ang = -dir*np.pi/2
        return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt

    def redraw(self,new=False):
        if new:
            self.poly_asset.set_xy(self.newGeometry)
            self.poly_asset.set_facecolor(self.color)

            self.poly_bb.set_xy(self.newBB)
            self.poly_bb.set_color(self.color)

            self.CP.set_offsets(np.c_[self.newCenter[0], self.newCenter[1]])

        else:
            self.poly_asset.set_xy(self.geometry)
            self.poly_asset.set_facecolor(self.color)

            self.poly_bb.set_xy(self.bounding_box)
            self.poly_bb.set_color(self.color)

            self.CP.set_offsets(np.c_[self.center[0], self.center[1]])
        self.poly_asset.figure.canvas.draw()



    def asset_bb(self,state):
        if state==1 or state=="on": self.poly_bb.set_alpha(1)
        elif state == 0 or state == "off": self.poly_bb.set_alpha(0)
        else: print(f'invalid input in DraggablePolygon.asset_bb({state})')

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        self.asset_bb("off")
        if event.inaxes != self.poly_asset.axes:return
        if self.lock: return
        contains, attrd = self.poly_asset.contains(event)
        if not contains: return
        self.asset_bb("on")

        if not self.newGeometry:x0, y0 = self.geometry[0]
        else:x0, y0 = self.newGeometry[0]
        self.press = x0, y0, event.xdata, event.ydata
        self.lock = True

    def update_loc(self,event):
        x0, y0, xpress, ypress = self.press
        dx = floor(event.xdata - xpress)
        dy = floor(event.ydata - ypress)

        # Update poly data --------------------------------
        xdx = [i + dx for i, _ in self.geometry]
        ydy = [i + dy for _, i in self.geometry]
        newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        # Update Center Point data ------------------------
        newCenter = self.center + np.array([dx, dy])
        # Update Bounding Box data ------------------------
        xdx = [i + dx for i, _ in self.bounding_box]
        ydy = [i + dy for _, i in self.bounding_box]
        newBB = [[a, b] for a, b in zip(xdx, ydy)]
        return newCenter,newGeometry,newBB

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        # Get cursor movement -----------------------------
        if not self.lock: return
        if event.inaxes != self.poly_asset.axes: return
        self.newCenter,self.newGeometry,self.newBB = self.update_loc(event)
        self.asset_bb("on")
        self.redraw(new=True)


    def on_release(self, event):
        'on release we reset the press data'
        if not self.lock:return
        self.press = None
        self.lock = False
        self.redraw()
        # self.geometry = self.newGeometry
        # self.center = self.newCenter
        # self.bounding_box = self.newBB
        # self.poly_asset.figure.canvas.draw()
        #print(f'state={self.center-0.5}')


    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidmotion)

if __name__=="__main__":
    main()
