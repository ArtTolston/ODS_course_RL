import gym
import gym_maze
import numpy as np
import time
import random
import scipy.stats as sps

from RandomAgent import RandomAgent
from CrossEntropyAgent import CrossEntropyAgent




def get_trajectory(agent, env, max_len=1000, transformer=None, visualize=False):
	trajectory = {'states': [], 'actions': [], 'rewards': []}

	obs = env.reset()
	state = agent.get_state(obs)


	for i in range(max_len):
		trajectory['states'].append(state)

		action = agent.get_action(state)
		trajectory['actions'].append(action)

		if transformer:
			next_obs, reward, done, _ = env.step(transformer[action])
		else:
			next_obs, reward, done, _ = env.step(action)
		trajectory['rewards'].append(reward)

		if done:
			return trajectory

		if visualize:
			print(f'state: {state}, action: {action}, reward: {reward}')
			env.render()
			time.sleep(0.1)

		state = agent.get_state(next_obs)

	return trajectory







env = gym.make('maze-sample-5x5-v0')

agent = CrossEntropyAgent()

turns = ['N', 'E', 'W', 'S']

n_iterations = 10
k_samples = 10
q = 0.8

#обучение
for i in range(n_iterations):
	trajectories = [get_trajectory(agent, env, transformer=turns) for _ in range(k_samples)]

	#оценка модели
	total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
	print(f'mean reward on {i} iteration: {np.mean(total_rewards)}')

	elite_trajectories = []
	quantile = np.quantile(total_rewards, q)
	for trajectory in trajectories:
		if np.sum(trajectory['rewards']) > quantile:
			elite_trajectories.append(trajectory)

	#улучшение модели
	agent.fit(elite_trajectories)



trajectory = get_trajectory(agent, env, transformer=turns, visualize=True)
print(f'total reward: {np.sum(trajectory["rewards"])}')
print(f'model: {agent.model}')
