import matplotlib.pyplot as plt
import matplotlib.patches as patches
#https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

class DraggablePolygon:
    lock = None
    def __init__(self):
        print('__init__')
        self.press = None

        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.geometry = [[0.0,0.0],[0.1,0.05],[0.2,0.15],[0.3,0.20],[0.4,0.25],[0.5,0.30],
                    [0.6,0.25],[0.7,0.15],[0.8,0.05],[0.9,0.025],[1.0,0.0]]
        self.newGeometry = []
        poly = plt.Polygon(self.geometry, closed=True, fill=False, linewidth=3, color='#F97306')
        ax.add_patch(poly)
        self.poly = poly

    def connect(self):
        'connect to all the events we need'
        print('connect')
        self.cidpress = self.poly.figure.canvas.mpl_connect(
        'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
        'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
        'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        print('on_press')
        if event.inaxes != self.poly.axes: return
        if DraggablePolygon.lock is not None: return
        contains, attrd = self.poly.contains(event)
        if not contains: return

        if not self.newGeometry:
            x0, y0 = self.geometry[0]
        else:
            x0, y0 = self.newGeometry[0]

        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggablePolygon.lock is not self:
            return
        if event.inaxes != self.poly.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        xdx = [i+dx for i,_ in self.geometry]
        ydy = [i+dy for _,i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly.set_xy(self.newGeometry)
        self.poly.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        print('on_release')
        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry


    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


dp = DraggablePolygon()
dp.connect()

plt.show()