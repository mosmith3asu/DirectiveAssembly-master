import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import floor


def main():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xticks(np.arange(1, 20))
    ax.set_yticks(np.arange(1, 20))

    center, geometry = np.array([1.5, 1.5]), [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]]
    dp1 = DraggablePolygon(fig,ax,geometry, center, 'r')
    dp1.connect()

    c_offset = np.array([0,5])
    dp2 = DraggablePolygon(fig,ax,geometry, center, 'orange')
    dp2.connect()

    plt.xlim([0, 20])
    plt.ylim([0, 20])

    plt.grid(True, markevery=1)
    plt.show()



rad    = lambda ang: ang*np.pi/180                 #lovely lambda: degree to radian
#https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

class DraggablePolygon:
    lock = None
    def __init__(self,ax,geometry,color,c_offset=[0,0]):
        """ Centered on [1.5,1.5] by default and
        parram: center: offset of block relative to [1.5,1.5]
        """
        print('__init__')
        self.press = None

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.set_xticks(np.arange(1,20))
        # ax.set_yticks(np.arange(1, 20))

        # L_2x2 ----------------------------------------
        #self.geometry = [[1, 1], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1]]
        #self.center = np.array([1.5, 1.5])
        #self.bounding_box = [[0,0],[3,0],[3,3],[0,3]]
        self.geometry = geometry
        self.center = np.array([1.5,1.5])+np.array(c_offset)
        cx,cy = self.center
        self.bounding_box = [[cx-1.5,cy-1.5],[cx+1.5,cy-1.5],[cx+1.5,cy+1.5],[cx-1.5,cy+1.5]]
        self.color = color


        # Draw Asset ----------------------------------
        self.CP = plt.scatter(self.center[0], self.center[1])
        self.poly_asset = plt.Polygon(self.geometry, closed=True, linewidth=1, facecolor=self.color , edgecolor=self.color )
        self.poly_bb = plt.Polygon(self.bounding_box, closed=True, fill=False, linestyle='--', linewidth=3,
                                   facecolor=self.color , edgecolor=self.color , alpha=0)
        # self.CP = plt.scatter(self.center[0], self.center[1])
        # self.poly_asset = plt.Polygon(self.geometry, closed=True, linewidth=3, facecolor='orange',edgecolor='k')
        # self.poly_bb= plt.Polygon(self.bounding_box , closed=True, fill=False,linestyle ='--',linewidth=3, facecolor='k', edgecolor='k', alpha=0)

        ax.add_patch(self.poly_asset)
        ax.add_patch(self.poly_bb)

        # Initi other vars ----------------------------
        self.newGeometry = []
        self.newCenter = []
        self.newBB = []


    def connect(self):
        'connect to all the events we need'
        print('connect')
        self.cidpress = self.poly_asset.figure.canvas.mpl_connect(
        'button_press_event', self.on_press)
        self.cidrelease = self.poly_asset.figure.canvas.mpl_connect(
        'button_release_event', self.on_release)
        self.cidmotion = self.poly_asset.figure.canvas.mpl_connect(
        'motion_notify_event', self.on_motion)
        self.keypress = self.poly_asset.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self,event):
        key, x0, y0 = event.key, event.xdata, event.ydata
        if event.inaxes != self.poly_asset.axes: return
        if DraggablePolygon.lock is not None: return
        if key == 'right': self.geometry = self.Rotate2D(self.geometry,self.center,dir=1)
        elif key == 'left': self.geometry = self.Rotate2D(self.geometry, self.center, dir=-1)
        #elif key=='enter':

        print('on_key:', key, self.center)
        self.poly_asset.set_xy(self.geometry)
        self.poly_asset.figure.canvas.draw()

    def Rotate2D(self,pts, cnt, dir = 1):
        ang = -dir*np.pi/2
        return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        #print('on_press')
        if event.inaxes != self.poly_asset.axes: return
        if DraggablePolygon.lock is not None: return
        contains, attrd = self.poly_asset.contains(event)
        if not contains: return
        if not self.newGeometry:x0, y0 = self.geometry[0]
        else:x0, y0 = self.newGeometry[0]
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

if __name__=="__main__":
    main()
