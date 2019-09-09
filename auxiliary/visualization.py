import _thread as thread
import visdom
import os


class Visualizer(object):
    def __init__(self, port, env):
        super(Visualizer, self).__init__()
        thread.start_new_thread(os.system, (f"visdom -p {port} > /dev/null 2>&1",))
        vis = visdom.Visdom(port=port, env=env)
        self.vis = vis

    def show_pointclouds(self, points, title=None, Y=None):
        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=dict(title=title, markersize=2))
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=dict(title=title, markersize=2)
            )
