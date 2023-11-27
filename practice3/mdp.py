from Frozen_Lake import FrozenLakeEnv
import numpy as np

import time

import matplotlib.pyplot as plt

state_n = 16
action_n = 4

env = FrozenLakeEnv(map_name='8x8')


def init_values():
	return {state: 0 for state in env.get_all_states()}


def policy_evaluation_step(policy, values, gamma):
	q_values = get_q_values(values, gamma)
	reward = 0
	new_values = {}
	for state in env.get_all_states():
		new_values[state] = 0
		for action in env.get_possible_actions(state):
			new_values[state] += policy[state][action] * q_values[state][action]
	return new_values


def policy_evaluation(policy, gamma):
	init_value = init_values()
	curr_values = init_value

	N = 100
	for _ in range(N):
		curr_values = policy_evaluation_step(policy, curr_values, gamma)
	return curr_values


def get_q_values(values, gamma):
	q_values = {}
	for state in env.get_all_states():
		q_values[state] = {}
		for action in env.get_possible_actions(state):
			q_values[state][action] = 0
			for next_state in env.get_next_states(state, action):
				next_state_prob = env.get_transition_prob(state, action, next_state)
				q_values[state][action] += next_state_prob * (env.get_reward(state, action, next_state) + gamma * values[next_state])
	return q_values


def update_policy(q_values):
	new_policy = {}
	
	for state in env.get_all_states():
		new_policy[state] = {}
		max_action = 0
		max_value = -100
		for action, value in q_values[state].items():
			new_policy[state][action] = 0.0
			if value > max_value:
				max_action = action
				max_value = value
		new_policy[state][max_action] = 1.0
	return new_policy


def init_policy():
	init_policy = {}
	for state in env.get_all_states():
		init_policy[state] = {}
		length = len(env.get_possible_actions(state))
		for action in env.get_possible_actions(state):
			init_policy[state][action] = 1.0 / length
	return init_policy

def print_values(values):
	n = 8
	i = 0
	for state, value in values.items():
		if i != 0 and i % n == 0:
			print(end='\n')
		print(f'{state}: {round(value, 2)}, ', end='')
		i += 1
	print(end='\n\n')



def policy_iteration(init_policy,  gamma=1.0):
	curr_policy = init_policy
	prev_policy = 0
	i = 0
	N = 50
	while prev_policy != curr_policy and i < N:
		i += 1
		values = policy_evaluation(curr_policy, gamma)
		#print_values(values)
		q_values = get_q_values(values, gamma)

		prev_policy = curr_policy
		curr_policy = update_policy(q_values)

	return curr_policy



#START HERE
# init_policy = init_policy()

# policy = policy_iteration(init_policy, gamma=0.5)

# print(policy)


def play(policy):
	total_reward = 0
	state = env.reset()
	for _ in range(100):
	    action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
	    #print(env.get_next_states(state, action))
	    state, reward, done, _ = env.step(action)
	    
	    #time.sleep(1)
	    #env.render()
	    
	    total_reward += reward
	    
	    if done or env.is_terminal(state):
	        break

	return total_reward











#here


gammas = [0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
totals = []

for gamma in gammas:
	initial_policy = init_policy()
	policy = policy_iteration(initial_policy, gamma=0.5)
	total_reward = 0
	print(f'\n\ngamma={gamma}\n\n')
	for _ in range(1000):
		total_reward += play(policy)
	totals.append(total_reward)


cat_gammas = [f'g={gamma}' for gamma in gammas]

# width = 0.3
# fig, ax = plt.subplot()

# ax.bar(x=np.arange(len(cat_gammas)), totals, width, label=f'gammas')
# ax.set_xticks(x)
# ax.set_xticklabels(cat_gammas)
# ax.legend()
plt.bar(cat_gammas, totals)
plt.savefig(f'gammas.png')
plt.clf()

