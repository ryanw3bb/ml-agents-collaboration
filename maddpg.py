import numpy as np
from ddpg_agent import Agent

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agents = []
        for i in range(num_agents):
            agent = Agent(state_size, action_size, random_seed)
            self.agents.append(agent)
            agent.critic_local = self.agents[0].critic_local
            agent.critic_target = self.agents[0].critic_target
            agent.critic_optimizer = self.agents[0].critic_optimizer
            agent.actor_local = self.agents[0].actor_local
            agent.actor_target = self.agents[0].actor_target
            agent.actor_optimizer = self.agents[0].actor_optimizer
            agent.memory = self.agents[0].memory

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
