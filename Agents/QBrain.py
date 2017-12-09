import gym
from gym import envs
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from config import config, config_net
from MemoryBuffer import ReplayMemory
from network.CNN import CNN
from network.deepnet import Net



 
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
Tensor = FloatTensor

config = config
config_net = config_net


# named tuple representing a single transition in enviornment
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))







class QLearningAgent(object):
    """
    Agent implementing Deep Q-learning with experience replay
    """

    def __init__(self, observation_space, action_space, learnable=None, lr_scheduler= None, replay_memory_capacity=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learnable = learnable
        self.epoch_done = 0
        
        self.epoch_durations = []
        self.average_total_Q = []
        self.average_total_reward = []
        self.average_loss = []
        self.eps_decay = []

        #// initialize replay memory D to capacity N
        self.replay_memory = ReplayMemory(replay_memory_capacity if replay_memory_capacity is not None else config["replay_memory_capacity"])

        #// initialize Q network

        if self.learnable is not None:
            self.q = CNN(config["frame_size_h"], config["frame_size_w"], \
                         config["action_list_n"]) if self.learnable=='CNN' \
                    else \
                    Net(config_net["inputs_dim"], \
                        config["action_list_n"], config_net["hidden_dim"])
            
            self.q.float()    
            self.q.train(True)    
            self.loss =  torch.nn.MSELoss()  
            self.optimizer = optim.Adam(self.q.parameters(), lr=config["learning_rate"], \
                                        betas=(0.001, 0.9), weight_decay=0.0001)
            if lr_scheduler:
                    self.optimizer = lr_scheduler(self.optimizer, self.epoch_done + 1)
                    
        else:
            self.q = None
            
        
        
    def act_random(self):
        action = self.action_space.sample()
        return action, 0    


      
    def act_learn(self, observation, eps=None):
        """
        Epsilon greedy control off-policy strategy      
            Args: 
                observation[DoubleTensor]: 
            Return: 
                action[int64] : action picked
                value[float]: estimated Q value of corresponding action picked
        """
        
        if eps is None:
            eps= config["EPS_END"] + (config["EPS_START"]-config["EPS_END"]) * \
                math.exp(-1. *  self.epoch_done / config["EPS_DECAY"])
            eps_decay.append(eps)

        # Compute Q-values for each action,
        # pick the action that maximizes Q or random action 
        observation = observation.view(config_net["inputs_dim"]) if self.learnable=='CNN' \
                     else observation.view(1, config_net["inputs_dim"])
        qvalue_list = self.q(Variable(observation, volatile=True).type(FloatTensor)) 
        
       
        
        if np.random.random() > eps:
            qvalue, action = torch.max(qvalue_list.data, 1)
            action = action.numpy()[0][0]   #from tensor 
            print "========= Q ACTION PICKING "
            print "Q(s) to pick from: "+str(qvalue_list)
            print "Agent pick the best one: " + str(action)
            print "of Q-value: "+ str(qvalue)
            print "========= \n"
        else: 
            qvalue, _= torch.max(qvalue_list.data, 1)
            action, _= self.act_random()
        
        qvalue = qvalue.numpy()[0][0]
        return action, qvalue           # always return MAX Q possible in current prediction 11/9

    def train(self, env):
        self.epoch_done=0
        t_num=0
        for i in range(config["n_itr"]):
            
            print('')
            print(' =============================> Epoch ' + str(self.epoch_done))

            obs = env.reset()
            obs = self.obFixer(self.observation_space, obs)
            qvalue_total=0.
            loss_total=0.
            loss_data_=0.
            reward_total=0.     # for lunar land
            

            for t in count():
                print(' episode: ' + str(t+1))
                
                # Select and perform the action
                action, qvalue = self.act_random() if self.q is None \
                                    else self.act_learn(torch.from_numpy(obs))
                obs2, reward, done, info = env.step(action)
                obs2 = self.obFixer(self.observation_space, obs2)
                
                 
                if done: 
                    obs2 = None  
                _reward_ =0.
                if reward>0:
                    _reward_ = 1
                elif reward<0:
                    _reward_ = -1

                
                ## I'm guaranteeing half of good transitions  
                if (t_num%2==0 and reward >0) or (t_num%2!=0 and reward<0):
                    self.replay_memory.save(obs, action, _reward_, obs2)
                    t_num += 1
                          
                
                # Perform on step of optimization to Q net
                if self.q is not None:  
                    loss_data_ = self.optimize_model()

                    
                reward_total += reward
                qvalue_total += qvalue
                loss_total += loss_data_
                obs = obs2 
                
                env.render()

                if done:
                    print 'Good Land!' if reward > 0 else 'Crash!'
                    self.average_total_reward.append(reward_total/(t+1))
                    self.average_total_Q.append(qvalue_total/(t+1))
                    self.average_loss.append(loss_total/(t+1))
                    self.epoch_durations.append(t+1)
                    break


            self.epoch_done += 1

        print('Complete')
        plot_durations()
        env.render(close=True)
        env.close()


    def optimize_model(self):

        BATCH_SIZE = config["BATCH_SIZE"]
        ACTION_LIST = config["action_list_n"]
        INPUT_SIZE = config_net["inputs_dim"]
        if len(self.replay_memory) < BATCH_SIZE:
            return 0.0

        transitions = self.replay_memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        print ('sample batch from memory...\n')
        
              
        # Concatenate the batch elements of each
        # state, action , reward in every transition
        state_batch = Variable(torch.from_numpy(np.asarray(batch.state))).float()    ##.double()
        action_batch = Variable(torch.from_numpy(np.asarray(batch.action))).float() ##.double()
        reward_batch = Variable(torch.from_numpy(np.asarray(batch.reward))).float()  ##.double()

        self.optimizer.zero_grad()
        
        # Compute Q(s_t, a) by model
        # the model computes Q(s_t) and we select the columns
        # of actions taken in record
        state_values_ = self.q(\
                    state_batch if self.learnable=='CNN' else state_batch.view(-1,INPUT_SIZE)\
                    )
        ##############
        #print (" V(s_{t}, a): "+ str(state_values_))
        ##############
        ##############
        #print ("action batch: "+ str(action_batch))
        ##############
        

        state_action_values = state_values_[0].view(1,ACTION_LIST).index_select(1, action_batch[0].long())
        for i in range(len(action_batch)-1):
            state_action_values = torch.cat((state_action_values, \
                                    state_values_[i+1].view(1,ACTION_LIST).index_select(1, action_batch[i+1].long())),0)
        state_action_values = state_action_values.index_select(1, Variable(torch.LongTensor([0])))
         
        
        ##############
        print (" Q(s_{t}, a): "+ str(state_action_values))
        ##############

        
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,\
                                             batch.next_state)))
        # We don't want backprop expected action values, volatile shuts down requires-grad
        non_final_next_states = Variable (torch.cat([torch.from_numpy(x) \
                                                     for x in np.asarray(batch.state) \
                                           if x is not None]), volatile=True).float()        ##.double()
           
        # Compute state value V(s_{t+1}) by model for all next states,
        # then select the maxQ value of each:
        ## MAX(a)Q(s_{t+1}, a)
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor)).float()       ##.double()
        next_state_values[non_final_mask], _ = torch.max( self.q( \
            non_final_next_states if self.learnable=='CNN' \
                    else non_final_next_states.view(-1,INPUT_SIZE) \
            ), 1)   
        
        
        
        #############
        print("V(s_{t+1}): " + str(next_state_values))
        ############

        # Compute the expected Q values
        ## Bellman equation
        ## Q_{t+1}(s,a) = r + gamma * MAX(a')Q(s', a')
        expected_state_action_values =  reward_batch + (next_state_values * config["GAMMA"]) 
        
        #############
        print("r: "+str(reward_batch))
        print("y_{t}: " + str(expected_state_action_values))
        ############
        

        # Compute Mean-Square Error loss from Q and expected Q*
        ## loss(x, y) = 1/n sum|x-y|^2
        loss = self.loss(state_action_values, expected_state_action_values)
        
        ##############
        #print("LOSS: " + str( loss.data.numpy()[0]))
        ################
        
         
        loss.backward()
        self.optimizer.step()
        
        return loss.data.numpy()[0]

    
    
    def plot_durations():
        fig = plt.figure(figsize=(14,14))
        plt.clf()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        reward_t = self.average_total_reward
        q_t = self.average_total_Q
        loss_ = self.average_loss
        eps_ = self.eps_decay

        #ax1.plot([x*y for x, y in zip(reward_t, epoch_durations)])
        ax1.plot(reward_t)
        ax2.plot(q_t)
        ax3.plot(loss_)
        ax4.plot(epoch_durations)

        ax1.set_title('REWARD - Training '+ config["game_name"])
        ax2.set_title('Q VALUE - Training '+ config["game_name"])
        ax3.set_title('LOSS - Training ' + config["game_name"])
        ax4.set_title('Epoch Duration - Training ' + config["game_name"])
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax3.set_xlabel('Epoch')
        ax1.set_ylabel('Average Reward per Epoch')
        ax2.set_ylabel('Average Action value(Q) oer Episode')
        ax3.set_ylabel('Average Loss per Episode')
        ax4.set_ylabel('Number of Steps before crash')

        print "\nAverage Reward Obtained: " + str(sum(reward_t)/config["n_itr"])
        print "Average Steps Per Epoch: " + str(sum(epoch_durations)/config["n_itr"])


    def obFixer(observation_space,observation): 
        """
        Fixes observation ranges scale to [-1, 1], uses hyperbolic tangent for infinite values
            Args: 
                obsercation_space[nparray]
                observation
            Return:
                newObservation[nparray]
        """
        newObservation = []

        if observation_space.__class__ == gym.spaces.box.Box:
            for space in range(observation_space.shape[0]):
                high = observation_space.high[space]
                low = observation_space.low[space]
                if high == np.finfo(np.float32).max or high == float('Inf'):    #tanh to scale for inf
                    newObservation.append(math.tanh(observation[space]))

                else:                         # current value percentage% to the range high-low
                    dif = high - low 
                    percent = (observation[space]+abs(low))/dif
                    newObservation.append(((percent * 2)-1).tolist())

        return np.asarray(newObservation)


    
    
    
    
    
