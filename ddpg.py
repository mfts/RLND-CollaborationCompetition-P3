import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy

from model import Actor, Critic

class DDPGAgent:
    def __init__(self, config):
        '''
        ----------------------------------
        Parameters
        
        state_size:   # of states
        action_size:  # of actions
        buffer_size:  size of the memory buffer
        batch_size:   sample minibatch size
        num_agents:   # of agents
        seed:         seed for random
        gamma:        discount rate for future rewards
        tau:          interpolation factor for soft update of target network
        lr_actor:     learning rate of Actor
        lr_critic:    learning rate of Critic
        weight_decay: L2 weight decay
        update_every: update every 20 time steps
        num_updates:  number of updates to the network
        ----------------------------------
        '''
        
        self.action_size = config.action_size
        self.state_size = config.state_size
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.num_agents = config.num_agents
        self.gamma = config.gamma
        self.tau = config.tau
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.weight_decay = config.weight_decay
        self.update_every = config.update_every
        self.num_updates = config.num_updates
        self.t_step = 0
        self.seed = random.seed(config.seed)
        self.device = config.device
        
        # Actor network agent
        self.actor_local = Actor(config.state_size, config.action_size, config.seed).to(config.device)
        self.actor_target = Actor(config.state_size, config.action_size, config.seed).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)
        
        # Critic network agent
        self.critic_local = Critic(config.state_size, config.action_size, config.seed).to(config.device)
        self.critic_target = Critic(config.state_size, config.action_size, config.seed).to(config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, weight_decay=config.weight_decay)
        
        # Noise paramter
        self.noise = OUNoise(config.action_size, config.seed)
        
        # init target networks
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)
                
    def act(self, states, add_noise=True):
        '''
        Agent selects action based on current state and selected policy
        '''
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)
        
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        '''
        Agent updates policy and value parameters based on experiences (state, action, reward, next_state, done)
        
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        '''
        states, actions, rewards, next_states, dones = experiences
        
        #--------- update critic -----------------------#
        # get current Q
        Q_expected = self.critic_local(states, actions)
        # get next action
        next_actions = self.actor_target(next_states)
        # get Qsa_next
        Q_targets_next = self.critic_target(next_states, next_actions)
        # calculate target with reward and Qsa_next
        Q_targets = rewards + (gamma* Q_targets_next * (1-dones))
        
        # calculate loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        #--------- update actor ------------------------#
        # computer actor loss
        pred_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, pred_actions).mean()
        
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #---------- update target networks -------------#
        # update target network parameters
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        '''
        Update target network weights gradually with an interpolation rate of TAU
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    ''' Ornstein-Uhlenbeck process '''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        ''' reset to internal state to initial mu '''
        self.state = copy.copy(self.mu)
        
    def sample(self):
        ''' update internal state and return it as a noise sample '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state