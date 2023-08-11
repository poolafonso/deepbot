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
import os
import json
import numpy as np
import random
import time
import sys
import csv
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from env.environment_stage_2 import Env
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation


EPISODES = 500
MAX_STEPS = 500
local_name = 'dqn-models'
training = False

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.result = Float32MultiArray()

        self.load_model = True
        self.load_episode = 1500
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+'/save_model/stage_01/'+local_name+'/'+'episode_'+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+'/save_model/stage_01/'+local_name+'/'+'episode_'+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)


def save_data(E, R, T, C, G, A):
    timestr = time.strftime("%H:%M:%S")
    episode = E
    episode_reward = R
    time_out = T
    collisions = C
    goals = G
    average_reward = A

    file = open(dirPath + '/save_file/stage_01/dqn_stage01.csv', 'a+')
    writer = csv.writer(file)
    data = [timestr, episode, episode_reward, time_out, collisions, goals, average_reward]
    writer.writerow(data)
    file.close()

def save_test(T, E, C, G, TT):
    timestr = time.strftime("%H:%M:%S")
    timeout = T
    episode = E
    collision = C
    goal = G
    total_time = TT

    file = open(dirPath + '/save_file/stage_01/test_dqn_stage01.csv', 'a+')
    writer = csv.writer(file)
    data = [timestr, timeout, episode, collision, goal, total_time]
    writer.writerow(data)
    file.close()

if __name__ == '__main__':
    rospy.init_node('deepbot_dqn_stage_1')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 28
    action_size = 5

    env = Env(action_size)

    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    if training:

        for e in range(agent.load_episode + 1, EPISODES):
            done = False
            state = env.reset()
            score = 0
 
            for t in range(agent.episode_step):
                action = agent.getAction(state)

                next_state, reward, done = env.step(action)

                agent.appendMemory(state, action, reward, next_state, done)

                if len(agent.memory) >= agent.train_start:
                    if global_step <= agent.target_update:
                        agent.trainModel()
                    else:
                        agent.trainModel(True)

                score += reward
                state = next_state
                get_action.data = [action, score, reward]
                pub_get_action.publish(get_action)

                if t > MAX_STEPS:
                    rospy.loginfo("Time out.")
                    done = True

                if e % 10 == 0:
                    agent.model.save(agent.dirPath + str(e) + '.h5')
                    with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                        json.dump(param_dictionary, outfile)

                if done:
                    result.data = [score, np.max(agent.q_value)]
                    pub_result.publish(result)
                    agent.updateTargetModel()
                    scores.append(score)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                                  e, score, len(agent.memory), agent.epsilon, h, m, s)
                    param_keys = ['epsilon']
                    param_values = [agent.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                global_step += 1
                if global_step % agent.target_update == 0:
                    rospy.loginfo("UPDATE TARGET NETWORK")

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

    #start test        
    else:

        for e in range(agent.load_episode + 1, EPISODES):
            done = False
            state = env.reset()
            score = 0
            c_timeout = 0
            c_goal = 0
            c_collision = 0
            start_step = time.time()
            
            for t in range(agent.episode_step):
                action = agent.getAction(state)

                next_state, reward, done = env.step(action)

                score += reward
                state = next_state
                get_action.data = [action, score, reward]
                pub_get_action.publish(get_action)

                if t >= MAX_STEPS:
                    c_timeout += 1
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    if t < MAX_STEPS:
                        c_collision += 1
                    break

                if reward >= 200:
                    rospy.loginfo("Sucess!!")
                    print("Episode:" + str(e))
                    print("Total Steps:" + str(t))
                    c_goal = 1
                    break

                end_step = time.time()
                time_step = end_step - start_step

            save_test(c_timeout, e, c_collision, c_goal, time_step)

