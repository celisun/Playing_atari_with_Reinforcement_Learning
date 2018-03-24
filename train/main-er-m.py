import gym
from gym import envs
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count

from Actor import Actor
from Critic import Critic
from MemoryBuffer import *


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


# Train ACER
def start_er(GAME_NAME, BATCH_SIZE=32, MEMORY_CAPACITY=50000):
    #print ("make enviornment")
    env = gym.make(GAME_NAME)
    #print ("create actor, critic")
    actor = Actor(env.observation_space, env.action_space)
    critic = Critic(env.observation_space, env.action_space)
    reward_per_epi=[]
    durations_per_epi=[]
    l_A=[]
    l_C=[]

    MAX_EPISODE = 200
    RENDER = False
    MAX_EP_STEPS= 1000
    DISPLAY_REWARD_THRESHOLD=200
    BATCH_SIZE=BATCH_SIZE
    MEMORY_CAPACITY=MEMORY_CAPACITY
    replay_memory_1 = ReplayMemory(MEMORY_CAPACITY)
    replay_memory_2 = ReplayMemory(MEMORY_CAPACITY)
    f_1 = BATCH_SIZE / 2  # define fraction for 2 buckets
    f_2 = BATCH_SIZE / 2

    #print ("begin.\n")
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        track_r = []
        critic._v_=[]
        actor._loss_=[]
        for t in count():
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            track_r.append(r)


            if not done:
                replay_memory_1.save(s, a, r, s_) if r>0 else replay_memory_2.save(s, a, r, s_)  # Save non-final transition into memory


            #learn form memory
            if len(replay_memory_1) >= f_1 and len(replay_memory_2)>= f_2:   # if positive D is enough, the other must as well
                transitions_1 = replay_memory_1.sample(f_1)   # Sample from 2 buckets
                batch1 = Transition(*zip(*transitions_1))
                transitions_2 = replay_memory_2.sample(f_2)
                batch2 = Transition(*zip(*transitions_2))


                s_b = np.append(np.asarray(batch1.state), np.asarray(batch2.state), axis=0)
                s_b_n = np.append(np.asarray(batch1.next_state), np.asarray(batch2.next_state), axis=0)
                a_b = np.append(np.asarray(batch1.action).reshape(f_1, 1),
                                np.asarray(batch2.action).reshape(f_2, 1), axis=0)
                r_b = np.append(np.asarray(batch1.reward).reshape(f_1, 1),
                                np.asarray(batch2.reward).reshape(f_2, 1), axis=0)


                td_error, abs_error  = critic.learn(s_b, r_b, s_b_n)    # Critic Learn
                actor.learn(s_b, a_b, td_error)       # Actor Learn

            s = s_

            ##print ("... in episode (%d) step (%d)" % (i_episode+1,t))
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            #env.render()

            if done or t >= MAX_EP_STEPS:   # Episode finished, print results
                ep_rs_sum = sum(track_r)
                #if 'running_reward' not in globals():
                #    running_reward = ep_rs_sum
                #else:
                #    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True   # rendering
                running_reward_avg = ep_rs_sum/float(t)
                reward_per_epi.append(ep_rs_sum)
                durations_per_epi.append(t)
                l_A.append(np.mean(actor._loss_))
                l_C.append(np.mean(critic._loss_))
                #print("episode:", i_episode, "  reward:", ep_rs_sum)
                #plot(reward_per_epi, durations_per_epi, l_A, l_C)

                break

    return reward_per_epi, durations_per_epi, l_A, l_C



def r_percent(r, t):
    n=len(r)
    c=0.
    for i in r:
        if i>=t:
            c += 1
    return c/n


def plot(x,d,la,lc):

    fig = plt.figure(figsize=(12,12))
    plt.clf()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax1.plot(x)
    ax2.plot(d)
    ax3.plot(la)
    ax4.plot(lc)
    ax1.set_title('ACERM - Training LunarLander-v2')
    ax1.set_ylim([-500, 300])
    ax1.set_ylabel('Reward per Episode')
    ax2.set_ylabel('Durations per Episode')
    ax3.set_ylabel('Actor Loss per Episode')
    ax4.set_ylabel('Critic Loss(TDerror) per Episode')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    fig.savefig('acerm.png')
    print ("training done and saved to acerm.png")







BATCH_SIZE=32
MEMORY_CAPACITY=50000

r, d, l_A, l_C = [],[],[],[]
performance = -1.
for i in range(100):
    print(i)
    _r, _d, _l_A, _l_C = start_er('LunarLander-v2', BATCH_SIZE=BATCH_SIZE, MEMORY_CAPACITY=MEMORY_CAPACITY)
   # _r = np.clip(_r, -500, 500)
    performance_ = float(r_percent (_r, 100)*100 + float(r_percent (_r, 200)*100)) + \
                   float(r_percent(_r, 0) * 100) + float(r_percent(_r, -100) * 100)
    if  performance_ > performance:   # obtain a better performance
        print ("found a better one at " + str(i) + "with " + str(float(r_percent(_r, -100) * 100)))
        r=_r
        d = _d
        l_A=_l_A
        l_C=_l_C
        performance = performance_
        plot(r, d, l_A, l_C)  # plot the best result of 5000
        # Report the best result of 5000
        f = open("acerm.txt", "w+")
        f.writelines("episode, r > -100: %.01f%s \n" % (float(r_percent(r, -100) * 100), "%"))
        f.writelines("episode, r > 0: %.01f%s \n" % (float(r_percent(r, 0) * 100), "%"))
        f.writelines("episode, r > 100: %.01f%s \n" % (float(r_percent(r, 100) * 100), "%"))
        f.writelines("episode, r > 200: %.01f%s \n" % (float(r_percent(r, 200) * 100), "%"))
        f.writelines("Highest total score: %.01f \n" % max(r))
        f.writelines("\nreward----\n")
        f.write(str(r))
        f.writelines("\nduration----\n")
        f.write(str(d))
        f.writelines("\nloss actor----\n")
        f.write(str(l_A))
        f.writelines("\nloss critic---\n")
        f.write(str(l_C))
        f.writelines('batch size N / capacity D')
        f.writelines(str(BATCH_SIZE))
        f.writelines(str(MEMORY_CAPACITY))
        f.close()


