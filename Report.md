
[//]: # (Image References)

[image1]: ./img/network.png "Network"
[image2]: ./img/untrained.gif "Untrained Agent"
[image3]: ./img/trained.gif "Trained Agent"
[image4]: ./img/scores.png "DDPG Scores"
[image5]: ./img/try1.png "DDPG Scores Try #1"
[image6]: ./img/try2.png "DDPG Scores Try #2"

# Report

## Introduction
The project consists of three programming files and a pre-built Unity-ML Agents environment.

The goal of the project is to train a pair of identical agents to play tennis with each other. This is achieved through reinforcement learning. The agent receives a reward of +0.1 for every time it is able to hit the ball over the net. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
The environment in which the agent can act is large (`n=24`); it contains position and velocity of the agents and ball stacked up for 3 sequences. The agent may can move the with 2 actions racket forward/backward and up/down. Each action is a continuous variable and can vary from -1 to 1. 
Therefore, we can use neither a traditional Q-Table approach (due to high-dimensional observation spaces) nor a Deep Q-Network [DQN] approach, which can solve high-dimensional observation spaces but only discrete, low-dimensional action spaces. In order to get around this barrier, we resort to a model-free, off-policy actor-critic algorithm that uses function approximation to learn policies in high-dimensional, continuous action spaces. This algorithm is call deep deterministic policy gradient [DDPG] algorithm. 

We deal with two cooperating agents as their shared interest is to keep the ball in play. Hence, we can modify the DDPG algorithm to allow for multiple agents.

The agents are trained successfully to earn an average cumulative reward of +0.5 over 100 episodes after 1458 episodes. In the next section, I will explain the learning algorithm used.

## Learning Algorithm
The DDPG algorithm used in the project consists of an **actor-critic network** combined with two familiar features from DQN: **experienced replay** and **gradual Q-target updates**.

At the heart of the algorithm is the actor-critic model, which is made up of two networks that aim to *learn the optimal policy* (**actor**) and *evaluate an action by computing the value function* (**critic**).

The **actor** neural network consists of 1 input layer, 1 hidden layer and 1 output layer. All layers are fully connected linear layers and map the observation space (states) to action space (actions). The network takes an input of 24 and expands the network to 512 nodes, then contracts to 256 nodes before returning 2 nodes, one for each action. There is a batch normalization layer after the first layer and between each layer there is also a ReLU activation function. The final output layer was a tanh layer to bound the actions.

The **critic** neural network consists of 1 input layer, 1 hidden layer and 1 output layer. All layers are fully connected linear layers and map state-action pairs to Q values. The network takes an input of 24 and expands the network to 512 nodes, then contracts to 256 nodes before returning 1 node, the Q value for a given state-action pair. There is a batch normalization layer after the first layer and between each layer there is also a ReLU activation function. Actions were not included until the second hidden layer. 

Both of the networks' layers were initialized from a uniform distribution, with the output layers initialized uniformly close to zero.

![actor-critic network][image1]

For the remainder the setup is similar to vanilla DQN. The algorithm is set up by initializing two identical Q-networks for the actor and critic each, for current and targeted Q-network weights. 
Unlike in the vanilla DQN, a shared replay buffer is used for both agents, to draw on experiences they may not even have encountered themselves. Because the agents goal is to cooperate and have the same action space, it makes sense to share these experiences.

Each agent selects an action based on the learned optimal policy of its local actor network.

At each time step, i.e. with each action taken, the agents update the replay buffer with the values for the current state, reward, action, next state, and whether the episode has terminated.
I have chosen to draw past replays of the agents every 4th action – there are 2 actions in total – and update each agent's target network based on the sampled experiences.

Since we have separated the optimal policy and the evaluation of actions in regard to Q values, we have to update *how the agents learn*. 

The **critic (or value) network learns** as follows, by
1. evaluating the current Q-values from the local critic network given current state-action pairs;
2. getting the next actions based on the target actor network and next states;
3. evaluating the next Q-values from the target critic network given next state-action pairs;
4. calculating the TD-error;
5. minimizing the loss computed as the mean squared difference between current Q-values and TD-error; and
6. gradually updating the weights on the target critic network with a small interpolation parameter.

The **actor (or policy) network learns** as follows, by
1. predicting the actions based on the local actor network for the current states;
2. evaluating the average loss with the local critic network given the state and predicted action pairs;
3. minimizing the loss;
4. gradually updating the weights on the target actor network with a small interpolation parameter.  

### Parameter Selection
To faciliate the parameter selection and instantiating of agents, I have set up a configuration file `config.py` that holds all parameters and passes them to `MADDPG()` and `DDPGAgent()`  when creating new instances.

```
ACTOR/CRITIC NETWORK PARAMETERS
=================
STATE_SIZE = 24         # agent-environment observation space
ACTION_SIZE = 2         # agent's possible action
FC1_UNITS = 512         # first fully-connected layer of network
FC2_UNITS = 256         # second fully-connected layer of network
NUM_AGENTS = 2          # number of agents

GENERAL PARAMETERS
=================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0

EXPERIENCE REPLAY MEMORY PARAMETERS
=================
BUFFER_SIZE = int(1e6)   # size of the memory buffer
BATCH_SIZE = 256         # sample minibatch size
MEMORY = lambda: ReplayBuffer(config.action_size, config.buffer_size, config.batch_size, config.seed, config.device)

AGENT PARAMETERS
=================
GAMMA = 0.99             # discount rate for future rewards
TAU = 0.02               # interpolation factor for soft update of target network
LR_ACTOR = 1e-4          # learning rate of Actor
LR_CRITIC = 3e-4         # learning rate of Critic
WEIGHT_DECAY = 0         # L2 weight decay
UPDATE_EVERY = 4         # update every 4 time steps
NUM_UPDATES = 1          # number of updates to the network

TRAINING PARAMETERS
===================
n_episodes=4000          # max number of episodes to train agent
max_t=1000              # max number of steps agent is taking per episode
```

## Training with DDPG
The agents are trained with the previously described DDPG over 4000 episodes with max. 1000 actions per episode. The DDPG learns an optimal policy to select appropriate actions and evaluates those actions in regards to Q values for each agent.

Below you can see snippet randomly selected actions (**untrained** agent):

![untrained][image2]

*The agents are frantically moving their rackets back and forth caused by taking actions at random.*

Compared with a **trained** agent: 

![trained][image3]

*The agents are able to play tennis consciously, batting the ball over net to each other.*

![scores][image4]

*The distribution shows the rewards per episode for each all agents (in blue) and the cumulative average maximum reward for 100 consecutive episodes (in red) for all agents.*

The environment has physical limitation: the bat can only move up and back within a certain limit. Therefore, some episodes are lost even though the ball could be caught in a normal game. The currently trained algorithm tops out at a maximum reward of +2.5 per episode.

## Improvements
Further improvements to the algorithm include 
- deep distributed distributional deterministic policy gradient (D4PG)
- N-step returns for inferring velocities using differences between frames 
- prioritized experience replay

## Conclusion
The DDPG algorithm is a huge improvement in the space of off-policy actor-critic algorithm, in particular for high-dimensional, continuous action spaces. The algorithm took quite a bit of fiddling around with the hyperparameters to learn consistently (see below). However, the addition of cooperative agents opens up a whole lot of new possibilities and brings us closer to learning real-world scenarios.

Next I will try the extension of the project, which consists of two cooperating agents competing against two cooperating agents in the game of soccer. The observation space is significantly larger at 112, and 6 and 4 actions for striker and goalie of each agent pair, respectively. This will involve making changes to how the agents are instantiated, in particular because the two cooperating agents (striker and goalie) have different reward functions. 

### Training attempts
*Failed training attempt*

![try 1][image5]

*Failed training attempt*

![try 2][image6]

*Successful training*
![final][image4]