import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Network.deepnetsoftmax import Netsoftmax
from config.configA import configA


class Actor(object):
    """Policy Gradient Learning algorithm.
    
       Action based method: network estimate the probability of each action P(a),
       loss is log probability * td_error, 
       P(a) reinforced by rewards from RL environment.
       
       update over every epoch (v.s.per step)
    """
    
    
    def __init__(self, observation_space, action_space, lr_scheduler=None):
        self.q = Netsoftmax(observation_space.shape[0], \
                        configA["outputs_dim"], configA["hidden_dim"])
       
        self.optimizer = optim.Adam(self.q.parameters(), lr=configA["learning_rate"], \
                                        betas=configA["betas"], \
                        weight_decay=configA["weight_decay"] if configA["weight_decay"] else None)
        self.ep_obs=[]
        self.ep_as=[]
        self.ep_rs=[]
        self._loss_=[]

    
    def choose_action(self, observation):
        observation = Variable(torch.from_numpy(observation), \
                     volatile=True).float().view(1, configA["inputs_dim"])
        
        prob_weights = self.q(observation)       # train
        
        print "(pick)Actor: act prob"
        print prob_weights
        
        # pick action 
        prob_weights = prob_weights.data.numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) 
        
        return action 
    
  
    def learn(self, s, a, td):
        if len(s.shape)==1: s, a = s[np.newaxis, :], np.array([[a]])    # (N, x)
          
        # Model predict action probability  
        self.optimizer.zero_grad()
        acts_prob = self.q(Variable(torch.from_numpy(np.asarray(s))).float())  # shape (N, 4)
        
        # Compute Loss -log probability * td-error
        log_prob = torch.log(torch.cat([acts_prob[i,a[i][0]] for i in range(a.shape[0])],0))          
        loss = -torch.mean(torch.cat([torch.mul(log_prob[i], \
                           float(td[i][0])) for i in range(td.shape[0])],0))      
                  
        print "Actor: log prob * vt (loss)"
        print -loss 
        
        # Update Model 
        loss.backward()
        self.optimizer.step()
        
        loss =  loss.data.numpy()[0] 
        self._loss_.append(-loss)
        return -loss
        
        
    def plot():
        return
        
        
