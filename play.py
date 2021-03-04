from unityagents import UnityEnvironment
import numpy as np
import torch
from maddpg import MADDPG

env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

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

agents = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=2)
agents.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agents.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


def play(n_episodes=5):
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        agents.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        print('Agent won point with score: {}'.format(round(np.max(scores), 2)))


play()
env.close()
