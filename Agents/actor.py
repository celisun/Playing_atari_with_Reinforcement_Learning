import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from deepnetsoftmax import Netsoftmax
from configA import configA


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

    def reset(self):
        self._loss_=[]
        self.ep_as=[]
        self.ep_obs=[]
        self.ep_rs=[]



    def choose_action(self, observation):
    # Pick an action
        observation = Variable(torch.from_numpy(observation), \
                  volatile=True).float().view(1, configA["inputs_dim"])

        prob_weights = self.q(observation)  # train

        #print observation
        #print "(pick)Actor: act prob"
        #print prob_weights

        prob_weights = prob_weights.data.numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())


        return action


    def learn(self, s, a, td):
    # s, a produces direction of Gradient ascent
    # loss = log probability * td_error

        if len(s.shape)==1: s, a = s[np.newaxis, :], np.array([[a]])    # (N, x)

        self.optimizer.zero_grad()  # clean buffer

        acts_prob = self.q(Variable(torch.from_numpy(np.asarray(s))).float())  # train

        #print "(learn)Actor: act prob"
        #print acts_prob

        # Compute Loss
        # log_prob = torch.log(acts_prob[0, a])
        # loss = -torch.mean(torch.mul(log_prob, td))
        log_prob = torch.log(torch.cat([acts_prob[i,a[i][0]]for i in range(a.shape[0])],0))     # (N, )
        loss = -torch.mean(torch.cat([torch.mul(log_prob[i], float(td[i][0])) \
                          for i in range(td.shape[0])],0))    #   (1, )

        #print "Actor: log prob * vt"
        #print -loss

        # Optimize Model
        loss.backward()
        self.optimizer.step()

        loss =  loss.data.numpy()[0]
        self._loss_.append(-loss)
        return -loss


    def plot():
        return
