import numpy as np
import torch.optim as optim
import ddpg_agent as ddpg
from ddpg_agent import Agent, ReplayBuffer
from model import Actor, Critic

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(ddpg.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(ddpg.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ddpg.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(ddpg.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(ddpg.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=ddpg.LR_CRITIC, weight_decay=ddpg.WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, ddpg.BUFFER_SIZE, ddpg.BATCH_SIZE, random_seed)

        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = Agent(self, state_size, action_size, random_seed)
            self.agents.append(agent)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones, timestep):
        for i, agent in enumerate(self.agents):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], i, timestep)

    def act(self, states, add_noise=True):
        actions = []
        for i, agent in enumerate(self.agents):
            state = np.reshape(states[i], (1, states.shape[1]))
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions
