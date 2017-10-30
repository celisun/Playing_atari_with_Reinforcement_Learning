import gym
from gym import envs
from gym import wrappers


import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import argparse
import logging
import sys

import torch
import torch.optim as optim
from torch.autograd import Variable


from DQN import DQN




config = {
            "frame_size_h": 9,
            "frame_size_w": 9,
            "action_list_n": 83,
            "EPS_START": 1,
            "EPS_END": 0.1,
            "EPS_DECAY": 200,
            "BATCH_SIZE": 32,
            "GAMMA": 0.999,
            "learning_rate": 1e-4,
            "replay_memory_capacity": 5000,
            "n_itr" : 100

        }


# named tuple representing a single transition in enviornment
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))
epoch_durations = []
average_total_Q = []
average_total_reward = []



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =[]
        self.position = 0

    def save(self, *args):      # save a trasition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        self.position= (self.position+1) % self.capacity


    def sample(self, batch_size):       # select random batch of transitions from memory
        return np.random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(epoch_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())





class QLearningAgent(object):
    """
    Agent implementing Deep Q-learning with experience replay
    """

    def __init__(self, observation_space, action_space, learnable=False, replay_memory_capacity=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epoch_done = 0


        #// initialize replay memory D to capacity N
        self.replay_memory = ReplayMemory(replay_memory_capacity if replay_memory_capacity is not None else config["replay_memory_capacity"])

        #// initialize Q network

        if learnable:
            self.q = DQN(config["frame_size_h"],config["frame_size_w"],config["action_list_n"])                  #input size, size, output    # Go 9X9 : box(3,9,9) discrete(83)
            self.optimizer = optim.RMSprop(self.q.parameters())
            #self.optimizer = optim.SGD(self.q.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=0.0001)
        else:
            self.q = None





    def act_random(self, observation):
        return self.action_space.sample(), 0




    #  Epsilon greedy control off-policy strategy
    def act_learn(self, observation, eps=None):
        if eps is None:
            eps= config["EPS_END"] + (config["EPS_START"]-config["EPS_END"]) * \
                math.exp(-1. *  self.epoch_done / config["EPS_DECAY"])

        # Compute Q-values for each action
        # pick the action that maximizes Q
        qvalue_list = self.q(
            Variable(observation, volatile=True).type(FloatTensor)).data             \
            if np.random.random() > eps \
            else self.action_space.sample()
        qvalue = qvalue_list.max(1)
        action = np.argmax(qvalue_list)


        return action, qvalue





    def train(self, env):

        self.epoch_done=0

        for i in range(config["n_itr"]):
            print('')
            print(' ---------------------------- Epoch ' + str(self.epoch_done))

            obs = env.reset()
            reward_total=0.
            qvalue_total=0.


            for t in count():
                print(' - - - - - -> episode: ' + str(t+1))

                # Select and perform the action
                action, qvalue = self.act_random(obs) if self.q is None else self.act_learn(obs)
                print ('action: ' + str(action) + 'Q-value: ' + str(qvalue))

                obs2, reward, done, info = env.step(action)
                print (reward, done, info)


                obs=
                obs2=

                # Store transition into memory
                self.replay_memory.save( ,action, reward, )



                # Perform on step of optimization to Q net
                if self.q is not None:
                    self.optimize_model()


                reward_total += reward
                qvalue_total += qvalue
                obs = obs2

                if done:
                    average_total_reward.append(reward_total/(t+1))
                    average_total_Q.append(qvalue_total/(t+1))
                    epoch_durations.append(t+1)
                    plot_durations()
                    break


            self.epoch_done += 1

        print('Complete')
        env.render(close=True)
        env.close()
        plt.ioff()
        plt.show()






    def optimize_model(self):

        BATCH_SIZE = config["BATCH_SIZE"]
        if len(self.replay_memory) < BATCH_SIZE:
            return

        transitions = self.replay_memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        print ('sample batch from memory...\n')


        # Concatenate the batch elements of each
        # state, action , reward in every transition
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))


        # Compute Q(s_t, a) by model
        # the model computes Q(s_t) and we select the columns
        # of actions taken in record
        state_action_values = self.q(state_batch).gather(1, action_batch)
        print (" Q(s_{t}, a)"+ str(state_action_values))


        # We don't want backprop expected action values, volatile shuts down requires-grad
        non_final_next_states = Variable (torch.cat([x for x in batch.next_state
                                                 if x is not None]),
                                      volatile=True)


        # Compute state value V(s_{t+1}) by model for all next states,
        # then select the maxQ value of each:
        #
        ## MAX(a)Q(s_{t+1}, a)
        next_state_values = self.q(non_final_next_states).max(1)    #####[0]
        print("V(s_{t+1}): "+str(next_state_values))

        # Compute the expected Q values
        #
        ## Bellman equation
        ## Q_{t+1}(s,a) = r + gamma * MAX(a')Q(s', a')
        expected_state_action_values = (next_state_values * config["GAMMA"]) + reward_batch
        print(expected_state_action_values)


        # Compute Mean-Square Error loss from Q and expected Q*
        #
        ## loss(x, y) = 1/n sum|x-y|^2
        loss = torch.nn.MSELoss(state_action_values, expected_state_action_values)
        print(loss)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()






def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    print(envs.registry.all())

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default='Go9x9-v0')
    args = parser.parse_args()

    env = gym.make(args.target)
    agent = QLearningAgent(env.observation_space, env.action_space, learnable=False)

    agent.train(env)











if __name__ == '__main__':
    sys.exit(main())














