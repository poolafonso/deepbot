import matplotlib.pyplot as plt
import rospy
import tf
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_matrix
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualiser:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'ro')
        self.max_x = 300
        self.max_y = 300
    
    def plot_init(self):
        self.ax.set_xlim(0, self.max_x)
        self.ax.set_ylim(0, self.max_y)
        return self.ln  

    def update_plot(self, frame):
        # data_map = np.random.randint(60, size=(60, 60))
        # self.ax.matshow(data_map, cmap="Greys")
        self.ax.matshow(self.mapper.probability_map(), cmap="Greys")
        return self.ln
    
    def lidar_callback(self, scan):
        scan_parameters = [scan.angle_min, scan.angle_max, scan.angle_increment]
        scan_ranges = np.array(scan.ranges)
        self.mapper.update_map(self.pose, scan_ranges, scan.angle_min, scan.angle_increment)


rospy.init_node('lidar_visual_node')
vis = Visualiser()
sub = rospy.Subscriber('/scan', LaserScan, vis.lidar_callback)
ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)
plt.show(block=True) 