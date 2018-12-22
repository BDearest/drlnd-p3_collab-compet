import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent2 import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE=int(1e5)
BATCH_SIZE=256
GAMMA=0.99
TAU=1e-3
LR_ACTOR=1e-4 ##2e-4
LR_CRITIC=3e-4 ##2e-3
WEIGHT_DECAY=0

NOISE_WEIGHT=1.0
NOISE_END=0.1
NOISE_DECAY=.9999

TIMESTEPS=1
LEARNSTEPS=3

class MADDPG():
    def __init__(self,num_agents,state_sizes,action_sizes,random_seed):
        self.num_agents=num_agents
        self.random_seed=random_seed
        self.agents=self.create_agents(state_sizes,action_sizes,LR_ACTOR,LR_CRITIC)
        self.memory = ReplayBuffer(action_sizes[0], BUFFER_SIZE, BATCH_SIZE, self.random_seed)
        self.noise_weight=NOISE_WEIGHT
        
    def create_agents(self,state_sizes,action_sizes,LR_ACTOR,LR_CRITIC):
        agents=[]
        
        for a in range(self.num_agents):
            agents.append(Agent(state_size=state_sizes[a],action_size=action_sizes[a],random_seed=self.random_seed,lra=LR_ACTOR,lrc=LR_CRITIC))
        return agents

    def step(self,ts,states,actions,rewards,next_states,dones):
        for a in range(self.num_agents):
            self.memory.add(states[a],actions[a],rewards[a],next_states[a],dones[a])
            if ts % TIMESTEPS == 0:
                if len(self.memory) > BATCH_SIZE:
                    for l in range(LEARNSTEPS):
                        experiences=self.memory.sample()
                        self.agents[a].learn(experiences,GAMMA)
                        ##self.agents[a].learn(experiences,NOISE_WEIGHT,GAMMA)
                        self.noise_weight=max(NOISE_END,self.noise_weight*NOISE_DECAY)
    
class ReplayBuffer:
    """Initialize a ReplayBuffer object.
    Params
    ======
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size=action_size
        self.memory=deque(maxlen=buffer_size)
        self.batch_size=batch_size
        self.experience=namedtuple("Experience", field_names=["state","action","reward","next_state","done"])
        self.seed=random.seed(seed)

    def add(self,state,action,reward,next_state,done):
        e=self.experience(state,action,reward,next_state,done)
        self.memory.append(e)

    def sample(self):
        experiences=random.sample(self.memory, k=self.batch_size)

        states=torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions=torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards=torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states=torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones=torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
