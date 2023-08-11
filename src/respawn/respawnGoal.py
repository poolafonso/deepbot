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
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('deepbot/src/respawn',
                                                'deepbot/models/goal_box/model.sdf')

        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        #self.init_goal_x = 0.6
        INITIAL POSITION FOR TRAINING 
        self.init_goal_x = 3.0
        self.init_goal_y = 0.0

        # INITIAL POSITION FOR TESTING 
        #self.init_goal_x = 6.0
        #self.init_goal_y = 0.0

        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_01 = -2.0, -2.0
        self.obstacle_02 = -2.0, 2.0
        self.obstacle_03 = 2.0, 2.0
        self.obstacle_04 = 2.0, -2.0

        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 2:
            
            while position_check:
                goal_x = random.randrange(-38, 39) / 10.0
                goal_y = random.randrange(-38, 39) / 10.0
                if abs(goal_x - self.obstacle_01[0]) <= 1.0 and abs(goal_y - self.obstacle_01[1]) <= 1.0:
                    position_check = True
                elif abs(goal_x - self.obstacle_02[0]) <= 1.0 and abs(goal_y - self.obstacle_02[1]) <= 1.0:
                    position_check = True
                elif abs(goal_x - self.obstacle_03[0]) <= 1.0 and abs(goal_y - self.obstacle_03[1]) <= 1.0:
                    position_check = True
                elif abs(goal_x - self.obstacle_04[0]) <= 1.0 and abs(goal_y - self.obstacle_04[1]) <= 1.0:
                    position_check = True
                elif abs(goal_x - 0.0) <= 1.0 and abs(goal_y - 0.0) <= 1.0:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1.0 and abs(goal_y - self.last_goal_y) < 1.0:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y
            '''
            # values for testing in corridor world
            
            while position_check:

                goal_x_list = [6.0, 5.9]
                goal_y_list = [0.0, 0.0]

                self.index = random.randrange(0, 2)
                #print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]
            '''
        else:
            while position_check:

                goal_x_list = [-5.5, -3.8, -3.9, -2.0, 3.55, 5.0, 1.2, -3.0]
                goal_y_list = [-5.5, -0.0, -2.0, -6.0, 6.2, 6.5, 2.0, -5.9 ]

                self.index = random.randrange(0, 8)
                #print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
