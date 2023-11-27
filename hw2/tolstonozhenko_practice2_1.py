import gym
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as opt


class LunarLanderModel(nn.Module):
	def __init__(self, state_dim, action_n):
		super().__init__()
		self.state_dim = state_dim
		self.action_n = action_n

		self.fc1 = nn.Linear(state_dim, 50)
		torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
		self.rl1 = nn.ReLU()
		self.fc2 = nn.Linear(50, action_n)
		torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

		self.model = nn.Sequential(
			self.fc1,
			self.rl1,
			self.fc2
		)

		self.softmax = nn.Softmax(dim=0)
		self.optimizer = opt.Adam(self.parameters(), lr=1e-2)
		self.lambdaLR = lambda epoch: 0.92 ** epoch
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lambdaLR)
		self.loss_func = nn.CrossEntropyLoss()

		self.epoch = 0

		self.uniform_dist = np.ones(self.action_n) / self.action_n
		self.eps = 0.1
		self.init_eps = 0.1

	def forward(self, _input):
		return self.model(_input)

	def get_action(self, state):
		state = torch.FloatTensor(state)
		#print(state.shape)
		logits = self.forward(state)
		action_prob = self.softmax(logits).detach().numpy()

		actual_prob = (1 - self.eps) * action_prob + self.eps * self.uniform_dist
		diff = 1.0 - np.sum(actual_prob)
		ind = np.random.choice(self.action_n, p=self.uniform_dist)
		actual_prob[ind] += diff
		#actual_prob = action_prob
		return np.random.choice(self.action_n, p=actual_prob)

	def fit(self, elite_trajectories):
		states = []
		actions = []

		for trajectory in elite_trajectories:
			for state, action in zip(trajectory['states'], trajectory['actions']):
				states.append(state)
				actions.append(action)

		states = torch.tensor(np.array(states), dtype=torch.float)
		actions = torch.tensor(actions, dtype=torch.long)

		#print(f'state: {states[0]}')

		pred_actions = self.forward(states)

		#print(f'pred_actions: {pred_actions[0]}')
		#print(f'actions: {actions[0]}')

		loss = self.loss_func(pred_actions, actions)

		print(loss)

		loss.backward()

		self.optimizer.step()
		self.scheduler.step()
		self.optimizer.zero_grad()

		self.epoch += 1
		self.eps = self.init_eps * self.lambdaLR(self.epoch)

		print(self.loss_func(self.forward(states), actions))


def get_trajectory(agent, env, max_len=1000, visualize=False):
	trajectory = {'states': [], 'actions': [], 'rewards': []}

	state = env.reset()


	for i in range(max_len):
		trajectory['states'].append(state)

		action = agent.get_action(state)
		trajectory['actions'].append(action)

		next_state, reward, done, _ = env.step(action)
		trajectory['rewards'].append(reward)


		if visualize:
			print(f'iteration: {i}, reward: {reward}')
			env.render()
			time.sleep(0.06)

		if done:
			return trajectory

		state = next_state

	return trajectory


def teach_model(agent, env, n_iterations, k_samples, q):
	average_rewards = []
	# обучение
	for i in range(n_iterations):

		trajectories = [get_trajectory(agent, env) for _ in range(k_samples)]

		# оценка модели
		total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
		average_rewards.append(np.mean(total_rewards))
		print(f'mean reward on {i} iteration: {np.mean(total_rewards)}')
		print(f'max reward on {i} iteration: {np.max(total_rewards)}')

		elite_trajectories = []
		quantile = np.quantile(total_rewards, q)
		for trajectory in trajectories:
			if np.sum(trajectory['rewards']) > quantile:
				elite_trajectories.append(trajectory)

		# улучшение модели
		#t1 = time.time()
		agent.fit(elite_trajectories)
		#print(f'learning time on {i} iteration: {time.time() - t1}')

	return agent, average_rewards




#env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v1')


n_iterations = 40
k_samples = 20
q = 0.8

agent = LunarLanderModel(state_dim=4, action_n=2)

agent, rewards = teach_model(agent, env, n_iterations, k_samples, q)

plt.plot(rewards)

get_trajectory(agent, env, visualize=True)