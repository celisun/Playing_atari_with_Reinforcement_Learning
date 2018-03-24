import gym
from gym import envs
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Actor import Actor
from Critic import Critic

from itertools import count

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()





# classic AC
def start(GAME_NAME, MAX_EPISODE):
    env = gym.make(GAME_NAME)  # create enviornment
    actor = Actor(env.observation_space, env.action_space)   # create actor
    critic = Critic(env.observation_space, env.action_space) # create critic
    reward_per_epi=[]
    durations_per_epi=[]
    l_A=[]
    l_C=[]
    MAX_EPISODE = MAX_EPISODE
    RENDER = False
    MAX_EP_STEPS= 1000
    #DISPLAY_REWARD_THRESHOLD=200

    #print ("begin.\n\n")
    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        critic.reset()
        actor.reset()
        track_r = []
        for t in count():
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)
            #if done: r = -20             # Penalty if die
            track_r.append(r)

            td_error, abs_error = critic.learn(s, r, s_)   # Critic Learn
            actor.learn(s, a, td_error)        # Actor Learn

            s = s_

            #print ("... in episode (%d) step (%d)" % (i_episode+1,t))
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
    #fig = plt.figure()
    fig = plt.figure(figsize=(12,12))
    #plt.clf()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax1.plot(x)
    ax2.plot(d)
    ax3.plot(la)
    ax4.plot(lc)
    ax1.set_title('AC - Training LunarLander-v2')
    ax1.set_ylim([-500, 300])
    ax1.set_ylabel('Reward per Episode')
    ax2.set_ylabel('Durations per Episode')
    ax3.set_ylabel('Actor Loss per Episode')
    ax4.set_ylabel('Critic Loss(TDerror) per Episode')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    fig.savefig('ac.png')
    print (" --- training done and saved to ac.png")








r, d, l_A, l_C = [],[],[],[]
performance = -1.

for i in range(50):
    print (i)
    _r, _d, _l_A, _l_C = start('LunarLander-v2', 500)
    #_r = np.clip(_r, -500, 500)
    performance_ = (float(r_percent (_r, 100)*100 + float(r_percent (_r, 200)*100)))
    if  performance_ > performance:   # obtain a better performance
        print ("found a better one at " + str(i) + "with " + str(performance_))
        r=_r
        d = _d
        l_A=_l_A
        l_C=_l_C
        performance = performance_

        plot(r, d, l_A, l_C)  # plot current best
        f = open("ac.txt", "w+")
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


#plot(r, d, l_A, l_C)   # plot the best result of 1000
# Report the best result of 1000













