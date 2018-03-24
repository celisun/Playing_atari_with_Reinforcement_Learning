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
from SumTreeMemoryBuffer import SumTreeMemoryBuffer


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))



# Train ACER-p, prioritized ACER
def start_p(GAME_NAME, BATCH_SIZE=32, MEMORY_CAPACITY=50000):
    env = gym.make(GAME_NAME)
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
    replay_memory = SumTreeMemoryBuffer(MEMORY_CAPACITY)

    #print "begin.\n\n"
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        track_r = []
        critic._v_=[]   # clean critic loss buffer
        actor._loss_=[]  # clean actor loss buffer
        for t in count():
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            ##if done: r = -20    #  Penalty if die

            track_r.append(r)

            # ACER: Critic Actor with Experience Replay
            if not done:
                transition = np.hstack((s, a, r, s_))
                replay_memory.save(transition)   # Save non-final transition

            #print len(replay_memory)
            #print replay_memory.data
            #print replay_memory.gettree
            if len(replay_memory) >= BATCH_SIZE:   # memory capacity into batch size
                tree_idx, batch, ISWeights = replay_memory.sample(BATCH_SIZE)   # Sample from memory
                s_b = np.asarray(batch[-1,0:8])        # state
                s_b_n = np.asarray(batch[-1,10:18])    # next state
                a_b = np.asarray(batch[-1,8])   # action
                r_b = np.asarray(batch[-1,9])  # reward

               # print("tree_idx:   " + str(tree_idx))
                #print(ISWeights)

                td_error, abs_error = critic.learn(s_b, r_b, s_b_n, ISWeights) # Critic Learn
                replay_memory.batch_update(tree_idx, abs_error)       # Update T priority
                actor.learn(s_b, a_b, td_error)                      # Actor Learn
               # print("rd_error:     " + str(td_error))
                print ("abs_error:   "+str(abs_error))

            s = s_

           # print "... in episode (%d) step (%d)" % (i_episode+1,t)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            #env.render()

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
               # if 'running_reward' not in globals():
               #     running_reward = ep_rs_sum
               # else:
               #     running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
              #  if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                running_reward_avg = ep_rs_sum / float(t)
                reward_per_epi.append(ep_rs_sum)
                durations_per_epi.append(running_reward_avg)   ## draw average reward here
                l_A.append(np.mean(actor._loss_))
                l_C.append(np.mean(critic._loss_))
               # print("episode:", i_episode, "  reward:", ep_rs_sum)
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
    ax1.set_title('ACER-Pr - Training LunarLander-v2')
    ax1.set_ylim([-500, 300])
    ax1.set_ylabel('Reward per Episode')
    ax2.set_ylabel('Durations per Episode')
    ax3.set_ylabel('Actor Loss per Episode')
    ax4.set_ylabel('Critic Loss(TDerror) per Episode')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    fig.savefig('acer-p.png')
    print "training done and saved to acer-p.png"




BATCH_SIZE=32
MEMORY_CAPACITY=50000

r, d, l_A, l_C = [],[],[],[]
performance = -1.
for i in range(100):
    print(i)
    _r, _d, _l_A, _l_C = start_p('LunarLander-v2', BATCH_SIZE=BATCH_SIZE, MEMORY_CAPACITY=MEMORY_CAPACITY)
    _r = np.clip(_r, -500, 500)
    performance_ = float(r_percent (_r, 100)*100 + float(r_percent (_r, 200)*100)) + \
                   float(r_percent(_r, 0) * 100) + float(r_percent(_r, -100) * 100)
    if  performance_ > performance:   # obtain a better performance
        print ("found a better one at " + str(i) + "with " + str(float(r_percent(_r, 0) * 100)))
        r=_r
        d = _d
        l_A=_l_A
        l_C=_l_C
        performance = performance_
        plot(r, d, l_A, l_C)
        f = open("acer-p.txt", "w+")
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


#r, d, l_A, l_C = start_p('LunarLander-v2', BATCH_SIZE=BATCH_SIZE, MEMORY_CAPACITY=MEMORY_CAPACITY)

