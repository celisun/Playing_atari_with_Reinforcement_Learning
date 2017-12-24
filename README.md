# Playing Atari with Reinforcemen Learning



Table comparing the average and total scores per episode by training on game LunarLanderv.2.0. using different learning algorithms 

<img src=https://raw.githubusercontent.com/celisun/2017-18Playing_Atari_with_Reinforcement_Learning/master/results_table.png width="450">

Lunar Lander game 'LunarLander-v2'

<img src=https://raw.githubusercontent.com/celisun/2017-18Playing_Atari_with_Reinforcement_Learning/master/rLunarLanderv.2.0.png width="450">

**Game Discription**: Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

## Random 
random agent: average r = -2.5, duration = 80 steps/episode:

<img src="https://raw.githubusercontent.com/celisun/2017-18Playing_Atari_with_DeepQLearning/master/results-random.png" width="550">

## Learned
#### Deep Q agent:

watch the game play video at https://www.youtube.com/watch?v=D5ymUS7umPQ&t=2s

<img src="https://raw.githubusercontent.com/celisun/2017-18Playing_Atari_with_DeepQLearning/master/results-Q.png" width="550">

#### Actor-Critic agent (with experience reply):

<img src=https://raw.githubusercontent.com/celisun/2017-18Playing_Atari_with_Reinforcement_Learning/master/results-AC-replay.png width="550">


## Dependencies

* python2
* pytorch
* numpy
* matplotlib

