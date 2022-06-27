# Project 3 (Collaboration and Competition) for Udacity Deep Reinforcement Learning Nanodegree

The project 3 solution for Udacity Deep Reinforcement Learning nano degree.

# Run the code

* 1. download this repository
* 2. install the requirements in a separate Anaconda environment: `pip install -r requirements.txt`
* 3. run the solution file [**Tennis.ipynb**](https://github.com/hynpu/drlnd_p3_compet/blob/main/Tennis.ipynb)

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

![MADDPG code](https://github.com/hynpu/drlnd_p3_compet/blob/main/images/maddpg%20psudo%20code.png)

# Approach

The high level structure shows as the following, and the code under [**maddpg_agent.py**](https://github.com/hynpu/drlnd_p3_compet/blob/main/maddpg_agent.py) follows the diagram:

![MADDPG illustration](https://github.com/hynpu/drlnd_p3_compet/blob/main/images/Screen%20Shot%202022-06-26%20at%207.08.58%20PM.png)

## 1. The state and action space of this environment

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
## 2. Explore the environment by taking random actions

We then take some random actions based on the environment we created just now, and see how the agents perform (apparently it will be bad without learning)

    for i in range(1, 6):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)

            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

## 3. Implement the MADDPG algo to train the agent

The last step is to implement the MADDPG algorithm to trian the agents. The code can be found in [**maddpg_agent.py**](https://github.com/hynpu/drlnd_p3_compet/blob/main/maddpg_agent.py). However, I would like to mention several techniques to improve the speed and convergence:

  * Adjust the OU noise by adding decreasing factors, and related discussions can be found in this repo: [Udacity discuss channel](https://knowledge.udacity.com/questions/25366)

  * Change different discount factor GAMMA to see the performance. The agent does not need to see too far to predict its next movement. So slightly reduce the GAMMA value to focus more on the current states.



# Results:

The average rewards along with the traning process show as following:

![Results](https://github.com/hynpu/drlnd_p3_compet/blob/main/images/results.png)
