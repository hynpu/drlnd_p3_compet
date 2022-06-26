@@ -1,2 +1,49 @@
<<<<<<< HEAD
# drlnd_p3_collaborate
 The project 3 (Tennis environment) of the Deep reinforcement learning for Udacity Nanodegree
=======
# Project 3 (Collaboration and Competition) for Udacity Deep Reinforcement Learning Nanodegree

The project 3 solution for Udacity Deep Reinforcement Learning nano degree.

# Run the code

* 1. download this repository
* 2. install the requirements in a separate Anaconda environment: `pip install -r requirements.txt`
* 3. run the solution file [**Tennis.ipynb**](https://github.com/hynpu/drlnd_p2_reacher/blob/main/Continuous_Control.ipynb)

# Goal

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# DDPG in detail

This Youtube video explained DDPG in a very clean way, and it is highly recommend to watch through the video and get some basic understanding of DDPG: 

[![DDPG youtube video](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/youtube%20link.PNG)](https://www.youtube.com/watch?v=oydExwuuUCw)

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. A high-level DDPG structure looks the following, and you can see it has some DQN features like the replay buffer, critic network and so on. As mentioned earlier: computing the maximum over actions in the target is a challenge in continuous action spaces. DDPG deals with this by using a target policy network to compute an action which approximately maximizes $Q_{\phi_{\text{targ}}}$. The target policy network is found the same way as the target Q-function: by polyak averaging the policy parameters over the course of training.

Putting it all together, Q-learning in DDPG is performed by minimizing the following MSBE loss with stochastic gradient descent:

![DDPG illustration](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/ddpg%20eqn.png)

The below image shows the compasiron between DDPG and DQN. 

![DDPG vs DQN](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/dqn-ddpg.png)

## MADDPG

MADDPG, or Multi-agent DDPG, extends DDPG into a multi-agent policy gradient algorithm where decentralized agents learn a centralized critic based on the observations and actions of all agents. 

It leads to learned policies that only use local information (i.e. their own observations) at execution time, does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents, and is applicable not only to cooperative interaction but to competitive or mixed interaction involving both physical and communicative behavior. The critic is augmented with extra information about the policies of other agents, while the actor only has access to local information. After training is completed, only the local actors are used at execution phase, acting in a decentralized manner.

>>>>>>> df9c398f7ff0d9cf77a969859ef85926089a0350