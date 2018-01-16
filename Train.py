import gym
from gym import envs
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Actor import Actor
from Critic import Critic
from SumTreeMemoryBuffer import SumTreeMemoryBuffer

from collections import namedtuple
from itertools import count

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# Train ACER-p, prioritized ACER
def start_p(GAME_NAME, BATCH_SIZE=32, MEMORY_CAPACITY=50000):
    env = gym.make(GAME_NAME)
    actor = Actor(env.observation_space, env.action_space)
    critic = Critic(env.observation_space, env.action_space)
    reward_per_epi=[]
    durations_per_epi=[]
    l_A=[]
    l_C=[]

    MAX_EPISODE = 150
    RENDER = False
    MAX_EP_STEPS= 1000
    DISPLAY_REWARD_THRESHOLD=200
    BATCH_SIZE=BATCH_SIZE
    MEMORY_CAPACITY=MEMORY_CAPACITY
    replay_memory = SumTreeMemoryBuffer(MEMORY_CAPACITY)

    print "begin.\n\n"
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        track_r = []
        critic._v_=[]   # clean critic loss buffer
        actor._loss_=[]  # clean actor loss buffer
        for t in count():
            if RENDER: env.render()

            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)

            if done: r = -20    #  Penalty if die

            track_r.append(r)

            # ACER: Critic Actor with Experience Replay
            if not done:
                transition = np.hstack((s, a, r, s_))
                replay_memory.save(transition)   # Save non-final transition

            if len(replay_memory) >= MEMORY_CAPACITY:   
                tree_idx, batch, ISWeights = replay_memory.sample(BATCH_SIZE)   # Sample from memory  
                s_b = np.asarray(batch[-1,0:8])        # state
                s_b_n = np.asarray(batch[-1,10:18])    # next state
                a_b = np.asarray(batch[-1,8])   # action
                r_b = np.asarray(batch[-1,9])  # reward

                td_error, abs_error = critic.learn(s_b, r_b, s_b_n, ISWeights) # Critic Learn
                replay_memory.batch_update(tree_idx, abs_error)       # Update priority
                actor.learn(s_b, a_b, td_error)                      # Actor Learn

            s = s_

            print "... in episode (%d) step (%d)" % (i_episode+1,t)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            #env.render() # display game window

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)/float(t)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                reward_per_epi.append(running_reward)
                durations_per_epi.append(t)
                l_A.append(np.mean(actor._loss_))
                l_C.append(np.mean(critic._loss_))
                print("episode:", i_episode, "  reward:", running_reward)
                #plot(reward_per_epi, durations_per_epi, l_A, l_C)

                break
    return reward_per_epi, durations_per_epi, l_A, l_C

def plot (x,d,la,lc):
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
    ax1.set_title('REWARD - Training LunarLander-v2')
    ax1.set_ylabel('Reward per Episode')
    ax2.set_ylabel('Durations per Episode')
    ax3.set_ylabel('Actor Loss per Episode')
    ax4.set_ylabel('Critic Loss(TDerror) per Episode')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    fig.savefig('acerp.png')
    print "training done and saved to acerp.png"

def _r_percent_(r, t):  # calculate reward percentage
    n=len(r)
    c=0.
    for i in r:
        if i>=t:
            c += 1
    return c/n




# ------ Train -----
r3, d3, l_A3, l_C3 = start_p('LunarLander-v2', BATCH_SIZE=1, MEMORY_CAPACITY=5)
plot(r3, d3, l_A3, l_C3)
print "episode, r > 0: %.01f%s" % (float(_r_percent_(r3, 0)*100), "%")
print "episode, r > -1: %.01f%s" % (float(_r_percent_(r3, -1)*100) , "%")
print "episode, r > -2: %.01f%s" % (float(_r_percent_(r3, -2)*100) , "%")
print "Highest score: %.02f" % max(r3)
print "Highest total score: %.01f" % max([x*y for x, y in zip(r3, d3)])
print "----"
print r3, d3, l_A3, l_C3
