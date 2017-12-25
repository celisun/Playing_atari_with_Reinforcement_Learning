import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Network.CNN import CNN
from Network.deepnetsoftmax import Netsoftmax
from config import ConfigA


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
    ## Choose a from s
    ## 这个行为不再是用 Q value 来选定的, 而是用概率来选定. 概率即预测概率
    ## 即使不用 epsilon-greedy, 也具有一定的随机性 
    ## Return the action picked  
        observation = Variable(torch.from_numpy(observation), \
                           volatile=True).float().view(1, configA["inputs_dim"])
        prob_weights = self.q(observation)  # train
        
        print "(pick)Actor: act prob"
        print prob_weights
        
        # Pick action using the probability predicted
        prob_weights = prob_weights.data.numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel()) 
        
        return action 
    
  
    def learn(self, s, a, td):
    # Actor 想要最大化期望的 reward, 在 Actor Critic 算法中, 
    # 我们用 “比平时好多少” (TD error) 来当做 reward
    # s, a 用于产生 Gradient ascent 的方向,
    # td 来自 Critic, 用于告诉 Actor 这方向对不对.
        if len(s.shape)==1: 
            s, a = s[np.newaxis, :], np.array([[a]])    # (N, x)
          
        # Model predict action probability  
        self.optimizer.zero_grad()
        acts_prob = self.q(Variable(torch.from_numpy(np.asarray(s))).float())        # shape (N, 4)
        
        # Compute Loss
        log_prob = torch.log(torch.cat([acts_prob[i,a[i][0]] for i in range(a.shape[0])],0))                 # log 动作概率    shape (N, )         
        loss = -torch.mean(torch.cat([torch.mul(log_prob[i], float(td[i][0])) for i in range(td.shape[0])],0))       # log 概率 * TD 方向    shape (1, )
                  
        print "Actor: log prob * vt"
        print -loss 
        
        # Update Model 
        loss.backward()
        self.optimizer.step()
        
        loss =  loss.data.numpy()[0] 
        self._loss_.append(-loss)
        return -loss
                
    def plot():
        return
        
        
