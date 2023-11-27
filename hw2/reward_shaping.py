import numpy as np

def reward_shaping(state, new_state, reward, potential, gamma):
	new_reward = reward + gamma * potential(new_state) - potential(state)
	return new_reward

def potential(state):
	return 30.0 * (np.sin(3 * state[0]) - 1)
#помимо этого нужно использовать значение

def potential_square(state):
	return 10.0 * (state[0] + np.pi) ** 2
#помимо этого нужно использовать значение 

def potential_add(state):
	return 1.5 * np.sin(3 *state[0])
#помимо этого нужно использовать значение 

def reward_shaping_new(state, new_state, action, reward):
	if (new_state[0] - state[0]) > 0 and action > 0 or (new_state[0] - state[0]) < 0 and action < 0:
		reward += 0.5
	return reward
