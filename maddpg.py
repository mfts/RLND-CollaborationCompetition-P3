from ddpg import DDPGAgent
import numpy as np
import torch
import torch.nn.functional as F

class MADDPG:
    def __init__(self, config):
        
        self.config = config
        self.gamma = config.gamma
        self.memory = config.memory()
        self.batch_size = config.batch_size
        self.update_every = config.update_every
        self.num_updates = config.num_updates
        self.t_step = 0
        
        self.maddpg_agents = [DDPGAgent(config) for _ in range(config.num_agents)]
        
    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()
            
    def act(self, all_states):
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.maddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        '''
        Agents takes next step
        - save most recent environment event to ReplayBuffer for each agent
        - load random sample from memory to agent's policy and value network 10 times for every 20 time steps 
        ''' 
        for s,a,r,ns,d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s,a,r,ns,d)
            
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    for agent in self.maddpg_agents:
                        experiences = self.memory.sample()
                        agent.learn(experiences, self.gamma)