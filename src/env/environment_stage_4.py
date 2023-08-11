#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #


#The following code has been modified, the original code has been
#implemented and distributed by ROBOTIS, under the Apache license.
#Available at: https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, Range
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from respawn.respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

        self.sub_bbox = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.BoundingBoxesCallback)
        self.object_count = rospy.Subscriber('/darknet_ros/found_object', ObjectCount, self.DetectionCountCallback)

        self.data_detection = ObjectCount()
        self.data_bounding_boxes = BoundingBoxes()

        rospy.Subscriber('/range/frl', Range, self.range_frl_callback)
        rospy.Subscriber('/range/frr', Range, self.range_frr_callback)
        rospy.Subscriber('/range/rl', Range, self.range_rl_callback)
        rospy.Subscriber('/range/rr', Range, self.range_rr_callback)

        self.range_frl = Range()
        self.range_frr = Range()
        self.range_rl = Range()
        self.range_rr = Range()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        sonar_range = []
        heading = self.heading
        min_range = 0.15
        min_sonar_range = 0.15
        done = False
        human_detections = 0

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        #return num detections using darknet_ros 
        human_detections = self.GetDetections()

        range_rl = self.get_range_rl()
        range_rr = self.get_range_rr()
        range_frl = self.get_range_frl()
        range_frr = self.get_range_frr()

        sonar_range.append(range_rl)
        sonar_range.append(range_rr)
        sonar_range.append(range_frl)
        sonar_range.append(range_frr)

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        if min_range > min(scan_range) > 0:
            done = True

        if min_sonar_range > min(sonar_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle, human_detections], done

    def setReward(self, state, done, action):
        yaw_reward = []
        human_detections = state[-1]
        obstacle_min_range = state[-3]
        current_distance = state[-4]
        heading = state[-5]
        ob_reward = 0

        if human_detections > 0 and obstacle_min_range < 0.6:
            ob_reward = human_detections * -5
        elif human_detections == 0 and obstacle_min_range < 0.4:
            ob_reward = -5
        else:
            ob_reward = 0


        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        range_rl = None
        while range_rl is None:
            try:
                range_rl = rospy.wait_for_message('/range/rl', Range, timeout=5)
            except:
                pass

        range_rr = None
        while range_rr is None:
            try:
                range_rr = rospy.wait_for_message('/range/rr', Range, timeout=5)
            except:
                pass

        range_frl = None
        while range_frl is None:
            try:
                range_frl = rospy.wait_for_message('/range/frl', Range, timeout=5)
            except:
                pass

        range_frr = None
        while range_frr is None:
            try:
                range_frr = rospy.wait_for_message('/range/frr', Range, timeout=5)
            except:
                pass
        
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)



    def range_frl_callback(self, msg):
        self.range_frl = msg

    def get_range_frl(self):
        return self.range_frl.range


    def range_frr_callback(self, msg):
        self.range_frr = msg

    def get_range_frr(self):
        return self.range_frr.range

    
    def range_rl_callback(self, msg):
        self.range_rl = msg

    def get_range_rl(self):
        return self.range_rl.range


    def range_rr_callback(self, msg):
        self.range_rr = msg

    def get_range_rr(self):
        return self.range_rr.range

#=========================================================HUMAN DETECT

    def DetectionCountCallback(self, data):
        self.data_detection = data
        

    def GetDetections(self):
        rate = rospy.Rate(5) # ROS Rate at 5Hz
        return self.data_detection.count
        rate.sleep()

    def BoundingBoxesCallback(self, data):
        self.data_bounding_boxes = data

    def GetBBox(self):
        return self.data_bounding_boxes