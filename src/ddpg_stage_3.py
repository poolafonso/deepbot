#!/usr/bin/env python

import rospy
import os
import json
import random
import time
import sys
import csv
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from env.environment_stage_3 import Env

import torch
from torch import nn 
import torch.nn.functional as F 
import torch.optim as opt 

import random
from copy import copy, deepcopy
from collections import deque
import numpy as np
print("Using torch version: {}".format(torch.__version__))

from matplotlib import pyplot as plt
from IPython.display import clear_output

import argparse

BUFFER_SIZE=1000000
BATCH_SIZE=64
GAMMA=0.99
TAU=0.001       
LRA=0.00025      
LRC=0.0025       
H1=400   
H2=300   

MAX_EPISODES=1500   
MAX_STEPS=500     
buffer_start = 100 
epsilon = 1.0
epsilon_decay = 0.1 
PRINT_DATA_EVERY = 10 
PRINT_PLOT_EVERY = 100

training = False 
load_model = True 
local_name = 'training' 
load_episode = 1398 
episode_step = 1000

class replayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        s, a, r, t, s2 = map(np.stack, zip(*batch))

        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0

#set GPU for faster training
cuda = torch.cuda.is_available() #check for CUDA
device   = torch.device("cuda" if cuda else "cpu")

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=3e-3):
        super(Critic, self).__init__()
                
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())

        self.linear2 = nn.Linear(h1+action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],1))
        
        x = self.relu(x)
        x = self.linear3(x)
        
        return x
    

class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def load(local_name, model_episode):
    actor.load_state_dict(
        torch.load(
            dirPath + '/save_model/stage_03/'+local_name+'/checkpoint_actor_'+str(model_episode)+ '.pkl',
            map_location=torch.device(device)
        )
    )
    print("Load checkpoint Actor Model")

    critic.load_state_dict(
        torch.load(
            dirPath + '/save_model/stage_03/'+local_name+'/checkpoint_critic_'+str(model_episode)+ '.pkl',
            map_location=torch.device(device)
        )
    )
    print("Load checkpoint Critic Model")



def subplot(R, P, A, Q):
    r = list(zip(*R))
    p = list(zip(*P))
    a = list(zip(*A))
    q = list(zip(*Q))
    clear_output(wait=True)
        
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r') #row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b') #row=1, col=0
    ax[0, 1].plot(list(a[1]), list(a[0]), 'g') #row=0, col=1
    ax[1, 1].plot(list(q[1]), list(q[0]), 'k') #row=1, col=1
    ax[0, 0].title.set_text('Episode reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Average reward')
    ax[1, 1].title.set_text('Q loss')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(dirPath + '/save_plot/stage_03/'+timestr+'_plot.png')
    plt.close()
    #plt.show()

def save_data(E, R, T, C, G, A):
    timestr = time.strftime("%H:%M:%S")
    episode = E
    episode_reward = R
    time_out = T
    collisions = C
    goals = G
    average_reward = A

    file = open(dirPath + '/save_file/stage_03/training/stage03_training.csv', 'a+')
    writer = csv.writer(file)
    data = [timestr, episode, episode_reward, time_out, collisions, goals, average_reward]
    writer.writerow(data)
    file.close()

def save_test(T, E, C, G, S):
    timestr = time.strftime("%H:%M:%S")
    timeout = T
    episode = E
    collision = C
    goal = G
    step = S

    file = open(dirPath + '/save_file/stage_03/test_stage03b.csv', 'a+')
    writer = csv.writer(file)
    data = [timestr, timeout, episode, collision, goal, step]
    writer.writerow(data)
    file.close()


if __name__ == '__main__':

    rospy.init_node('deepbot_stage_3', anonymous=True, log_level=rospy.INFO, disable_signals=False)
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    #torch.manual_seed(-1)

    state_dim = 29
    action_dim = 5
    env = Env(action_dim)

    #print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    critic  = Critic(state_dim, action_dim).to(device)
    actor = Actor(state_dim, action_dim).to(device)

    target_critic  = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim).to(device)

    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
        
    q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)#, weight_decay=0.01)
    policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)

    MSE = nn.MSELoss()

    memory = replayBuffer(BUFFER_SIZE) 

    plot_reward = []
    plot_policy = []
    plot_q = []
    plot_steps = []
    plot_avg_reward = []

    scores, episodes = [], []

    best_reward = -np.inf
    saved_reward = -np.inf
    saved_ep = 0
    total_reward = 0
    avg_reward = 0
    global_step = 0

    if training:

        if load_model:
            load(local_name, load_episode)
        
        for episode in range(load_episode + 1, MAX_EPISODES):
            done = False
            state = deepcopy(env.reset())
            #print("State:" + str(state))
            noise.reset()

            ep_reward = 0
            ep_q_value = 0
            step=0
            c_timeout = 0
            c_goal = 0
            c_collision = 0       
            
            for step in range(episode_step):
                loss=0
                global_step +=1
                epsilon -= epsilon_decay
                #actor.eval()
                action = actor.get_action(state)
                #actor.train()
                legal_action = np.argmax(actor.get_action(state))
                #print("Step:" + str(step))

                action += noise()*max(0, epsilon)
                action = np.clip(action, -1.5, 1.5)
                
                next_state, reward, done = env.step(legal_action)


                memory.add(state, action, reward, done, next_state)

                #keep adding experiences to the memory until there are at least minibatch size samples
                
                if memory.count() > buffer_start:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(BATCH_SIZE)

                    s_batch = torch.FloatTensor(s_batch).to(device)
                    a_batch = torch.FloatTensor(a_batch).to(device)
                    r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                    t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
                    s2_batch = torch.FloatTensor(s2_batch).to(device)
                    
                    
                    #compute loss for critic
                    a2_batch = target_actor(s2_batch)
                    target_q = target_critic(s2_batch, a2_batch) #detach to avoid updating target
                    y = r_batch + (1.0 - t_batch) * GAMMA * target_q.detach()
                    q = critic(s_batch, a_batch)
                    
                    q_optimizer.zero_grad()
                    q_loss = MSE(q, y) #detach to avoid updating target
                    q_loss.backward()
                    q_optimizer.step()
                    
                    #compute loss for actor
                    policy_optimizer.zero_grad()
                    policy_loss = -critic(s_batch, actor(s_batch))
                    policy_loss = policy_loss.mean()
                    policy_loss.backward()
                    policy_optimizer.step()
                    
                    #soft update of the frozen target networks
                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - TAU) + param.data * TAU
                        )

                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - TAU) + param.data * TAU
                        )

                state = deepcopy(next_state)
                ep_reward += reward

                get_action.data = [legal_action, ep_reward, reward]
                pub_get_action.publish(get_action)
                    
                if step >= MAX_STEPS:
                    c_timeout += 1
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    noise.reset()           
                    scores.append(ep_reward)
                    episodes.append(episode)
                    if step < MAX_STEPS:
                        c_collision += 1
                    break

                if reward >= 200:
                    c_goal += 1
           
            try:
                plot_reward.append([ep_reward, episode+1])
                plot_policy.append([policy_loss.data, episode+1])
                plot_q.append([q_loss.data, episode+1])
                plot_steps.append([step+1, episode+1])

                avg_reward += np.mean(plot_reward)
                plot_avg_reward.append([avg_reward, episode+1])
            except:
                continue
            
            total_reward += ep_reward
            avg_reward = total_reward/episode

            result.data = [ep_reward, avg_reward]
            pub_result.publish(result)
              
            if ep_reward > best_reward:
                torch.save(actor.state_dict(), dirPath + '/save_model/stage_03/training/checkpoint_actor_'+str(episode)+'.pkl')
                torch.save(critic.state_dict(), dirPath + '/save_model/stage_03/training/checkpoint_critic_'+str(episode)+'.pkl')
                
                best_reward = ep_reward
                saved_reward = ep_reward
                saved_ep = episode

            print("Episode:" + str(episode))
            print("Ep Reward:" + str(ep_reward))
            print("Average Reward:" + str(avg_reward))
            print("Best reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))

            if (episode % PRINT_DATA_EVERY) == (PRINT_DATA_EVERY-1):    # print every print_every episodes
                print('[%6d episodes, %8d total steps] average reward for past {} iterations: %.3f'.format(PRINT_DATA_EVERY) %
                      (episode + 1, global_step, avg_reward))
                print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))

            if (episode % PRINT_PLOT_EVERY) == (PRINT_PLOT_EVERY-1):
                subplot(plot_reward, plot_policy, plot_avg_reward, plot_q)

            save_data(episode, ep_reward, c_timeout, c_collision, c_goal, avg_reward)

    #start test        
    else:
        
        load(local_name, load_episode)

        #done = False
        #state = deepcopy(env.reset())
        
        for episode in range(load_episode + 1, MAX_EPISODES):
            done = False
            state = deepcopy(env.reset())
            #print("State:" + str(state))
            #noise.reset()

            ep_reward = 0
            ep_q_value = 0
            step=0
            collision = 0
            c_goal = 0
            timeout = 0
            start_step = time.time()
            
            for step in range(episode_step):

                #loss=0
                global_step +=1
                epsilon -= epsilon_decay
                #actor.eval()
                action = actor.get_action(state)
                #actor.train()
                legal_action = np.argmax(actor.get_action(state))
                #print("Step:" + str(step))

                action += noise()*max(0, epsilon)
                action = np.clip(action, -1.5, 1.5)
                state, reward, done = env.step(legal_action)
                #print(reward)
                if step >= MAX_STEPS:
                    timeout += 1
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    if step < MAX_STEPS:
                        collision += 1
                    break

                if reward >= 200:
                    rospy.loginfo("Sucess!!")
                    print("Episode:" + str(episode))
                    print("Total Steps:" + str(step))
                    c_goal = 1
                    break    

                end_step = time.time()
                time_step = end_step - start_step  

            save_test(timeout, episode, collision, c_goal, time_step)
