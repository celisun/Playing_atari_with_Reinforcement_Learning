import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Network.CNN import CNN
from Network.deepnet import Net

from config import configC


class Critic(object):
    # Same as Q learning, but estimate state value, not state action value. 
    # Compute the loss between v and v_next, through TD_error = (r+gamma*V_next) - V_eval.  
    # Supports both single sample and a batch of samples for training  
    
    def __init__(self, observation_space, action_space, lr_scheduler=None): 
    # 搭建好 Critic 神经网络
    
        self.q = Net(observation_space.shape[0], configC["outputs_dim"], configC["hidden_dim"])
        self.optimizer = optim.Adam(self.q.parameters(), lr=configC["learning_rate"], betas=configC["betas"], \
                                 weight_decay=configC["weight_decay"] if configC["weight_decay"] else None)
        self.loss =  torch.nn.MSELoss()  
        self._loss_=[]
        
        
    def learn(self, s, r, s_):  
    
        if len(s.shape)==1: 
            s, s_, r = s_[np.newaxis, :], s_[np.newaxis, :], np.array([[r]]) 
        
        # Construct Vriable for pytorch
        s  = Variable(torch.from_numpy(np.asarray(s))).float()              # shape=(N, 8)
        s_ = Variable(torch.from_numpy(np.asarray(s_)), volatile=True).float()      # shape=(N, 8)
           
        self.optimizer.zero_grad()
    
        # Model estimate state values
        state_values = self.q(s)     # v shape=(N, 1)
    
        # Model estimate next state values 
        next_state_values = self.q(s_)     # v_ shape=(N, 1)
        next_state_values.volatile = False
        
        # Copmute expected state values 
        expected_state_values = torch.cat([torch.add((next_state_values[i] * configC["GAMMA"]), \
                            r[i][0]) for i in range(r.shape[0])],0).view(s.size()[0], 1)        # v_ex = gamma*v_ + r   
        # Copmute TD error and loss
        td_error = expected_state_values.sub(state_values)      # TD_error = r+gamma*v_ - V   shape=(N, 1)
        loss = torch.mean((expected_state_values.sub(state_values))**2)   # mean squared TD_error    shape=(1, )

        # Update model
        loss.backward()
        self.optimizer.step()
    
        # Append loss to array
        self._loss_.append(torch.mean(td_error).data.numpy()[0])
     
        td_error = td_error.data.numpy()    
        return td_error    
    
    def plot():
        return
           
