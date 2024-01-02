import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import floor

rad    = lambda ang: ang*np.pi/180                 #lovely lambda: degree to radian
#https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

class DraggablePolygon:
    lock = None
    def __init__(self):
        print('__init__')
        self.press = None

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xticks(np.arange(1,20))
        ax.set_yticks(np.arange(1, 20))
        # Tmp Input --------------------------------------
        self.block_def = ["L_2x2", "L_2x2", "L_3x2"]
        self.block_color = ['r','b','o']

        # Static Parameters---------------------------------
        self._poly_defs_= {
            "L_2x2": [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]],
            "L_3x2": [[1, 0], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]],
        }
        self._center_ = np.array([1.5, 1.5])
        self._bb_ = [[0,0],[3,0],[3,3],[0,3]]

        # Create Asset Lists ----------------------------------------
        self.geometry = []
        for name in self.block_def:
            self.geometry.append(self._poly_defs_[name])
            self.center.append(self._center_ )
            self.bounding_box.append(self._bb_)

        #self.geometry= [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]]
        #self.center = np.array([1.5, 1.5])
        #self.bounding_box = [[0,0],[3,0],[3,3],[0,3]]

        # Draw Assets ----------------------------------
        self.render_center = []
        self.render_asset = []
        for i, name in enumerate(self.block_def):
            self.render_center.append(plt.scatter(self.center[i][0], self.center[i][1]))
            self.render_asset.append(plt.Polygon(self.geometry[i], closed=True, linewidth=1, facecolor=self.block_color[i]))
            ax.add_patch(self.render_asset[i])

        # Initi other vars ----------------------------
        self.newGeometry = []
        self.newCenter = []
        self.ipoly = None




    def Rotate2D(self,pts, cnt, dir = 1):
        ang = -dir*np.pi/2
        return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt
    def check_block_selection(self,event):
        if DraggablePolygon.lock is not None: return
        for i, name in enumerate(self.block_def):
            if event.inaxes == self.render_asset[i].axes: return i
        return None # no block under cursor

    def redraw(self,ipoly):
        self.render_asset[ipoly].set_xy(self.geometry[ipoly])
        self.render_center[ipoly].set_offsets(np.c_[self.center[ipoly][0], self.center[ipoly][1]])
        self.poly_asset.figure.canvas.draw()
    def unpack_asset(self,ipoly):
        center, geometry = self.center[ipoly], self.geometry[ipoly]
        render_center,render_asset = self.render_center[ipoly],self.render_asset[ipoly]
        return center,geometry,render_center,render_asset

    # Interaction Functions ###########################################################
    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.poly_asset.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.poly_asset.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.poly_asset.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.keypress = self.poly_asset.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self,event):
        # Event Handler -----------------------
        key, x0, y0 = event.key, event.xdata, event.ydata
        if self.check_block_selection(event) is None: return

        # Performan Transformation -----------------------
        center,geometry,render_center,render_asset = self.unpack_asset(self.ipoly)
        if key == 'right': geometry = self.Rotate2D(geometry,center,dir=1)
        elif key == 'left': geometry = self.Rotate2D(geometry, center, dir=-1)
        #elif key=='enter':

        # Update Asset Data -----------------------------
        self.geometry[self.ipoly] = geometry
        self.center[self.ipoly] = center
        self.redraw(self.ipoly)





    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        # Event Handler -----------------------
        self.ipoly = self.check_block_selection(event)
        if self.ipoly is None: return
        #if event.inaxes != self.poly_asset.axes: return
        #if DraggablePolygon.lock is not None: return

        # Performan Transformation -----------------------
        center,geometry,render_center,render_asset = self.unpack_asset(self.ipoly)
        contains, attrd = render_asset[self.ipoly].contains(event)
        if not contains: return
        if not self.newGeometry: x0, y0 = geometry[0] # first new geometer therefore init
        else: x0, y0 = self.newGeometry[0] # else get new geometry
        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        # Get cursor movement -----------------------------
        if DraggablePolygon.lock is not self: return
        if event.inaxes != self.poly_asset.axes: return
        x0, y0, xpress, ypress = self.press
        dx = floor(event.xdata - xpress)
        dy = floor(event.ydata - ypress)

        # Update poly data --------------------------------
        xdx = [i+dx for i,_ in self.geometry]
        ydy = [i+dy for _,i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly_asset.set_xy(self.newGeometry)

        # Update Center Point data ------------------------
        self.newCenter = self.center+ np.array([dx,dy])
        self.CP.set_offsets(np.c_[self.newCenter[0], self.newCenter[1]])

        # Update Bounding Box data ------------------------
        xdx = [i + dx for i, _ in self.bounding_box]
        ydy = [i + dy for _, i in self.bounding_box]
        self.newBB = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly_bb.set_xy(self.newBB)
        self.poly_bb.set_alpha(1)

        # Update Plot -------------------------------------
        self.poly_asset.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        #print('on_release')
        if DraggablePolygon.lock is not self:return
        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry
        self.center = self.newCenter
        self.bounding_box = self.newBB
        self.poly_bb.set_alpha(0)
        self.poly_asset.figure.canvas.draw()

        print(f'state={self.center-0.5}')

    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly_asset.figure.canvas.mpl_disconnect(self.cidmotion)


dp = DraggablePolygon()
dp.connect()

plt.xlim([0,20])
plt.ylim([0,20])

plt.grid(True,markevery=1)
plt.show()