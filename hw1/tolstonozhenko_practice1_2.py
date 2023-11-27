import gym
import numpy as np
import time
import matplotlib.pyplot as plt

all_rewards = []

# в классе добавились новые атрибуты
# тип сглаживания: Лаплас или полиси
# значение альфы - параметра сглаживания
# Помимо этого слегка изменился метод fit. Добавлены случаи вычисления с разными типами сглаживания
class CrossEntropyAgent:
		def __init__(self, state_n=25, action_n=4, smoothing=None, alpha=None):
			self.state_n = state_n
			self.action_n = action_n
			self.model = np.ones((self.state_n, self.action_n), dtype=float) / self.action_n

			self.smoothing_types = ['laplace', 'policy']
			if smoothing is None:
				return
			elif smoothing in self.smoothing_types:
				self.smoothing = smoothing
			else:
				raise NotImplementedError()

			self.init_alpha = alpha
			self.alpha = alpha
			self.with_alpha_decay = False
			self.counter = 1


		def get_action(self, state):
			action = np.random.choice(self.action_n, p=self.model[state])
			return action

		def fit(self, elite_trajectories):
			new_model = np.zeros((self.state_n, self.action_n), dtype=float)

			for trajectory in elite_trajectories:
				for state, action in zip(trajectory['states'], trajectory['actions']):
					new_model[state][action] += 1

			if self.smoothing is None:
				for state in range(self.state_n):
					if np.sum(new_model[state]) > 0:
						new_model[state] /= np.sum(new_model[state])
					else:
						new_model[state] = self.model[state].copy()
				self.model = new_model
				
			elif self.smoothing == 'laplace':
				for state in range(self.state_n):
					new_model[state] = (new_model[state] + self.alpha) / (np.sum(new_model[state]) + self.alpha * self.action_n)
				self.model = new_model

			elif self.smoothing == 'policy':
				for state in range(self.state_n):
					if np.sum(new_model[state]) > 0:
						new_model[state] /= np.sum(new_model[state])
					else:
						new_model[state] = self.model[state].copy()
				self.model = self.alpha * new_model + (1 - self.alpha) * self.model

			if self.with_alpha_decay:
				self.alpha = self.init_alpha * np.exp(-0.01*self.counter)
				self.counter += 1

			return None


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
			print(f'iteration: {i}, state: {state}, action: {action}, reward: {reward}')
			env.render()
			time.sleep(0.02)

		if done:
			return trajectory

		state = next_state

	return trajectory


def teach_model(agent, env, n_iterations, k_samples, q):
	average_rewards = []
	#обучение
	for i in range(n_iterations):
		trajectories = [get_trajectory(agent, env) for _ in range(k_samples)]

		#оценка модели
		total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
		global all_rewards
		all_rewards.extend(total_rewards)
		average_rewards.append(np.mean(total_rewards))
		print(f'mean reward on {i} iteration: {np.mean(total_rewards)}', end='\r')

		elite_trajectories = []
		quantile = np.quantile(total_rewards, q)
		for trajectory in trajectories:
			if np.sum(trajectory['rewards']) > quantile:
				elite_trajectories.append(trajectory)

		#улучшение модели
		agent.fit(elite_trajectories)

	return agent, average_rewards



def find_optimal_alphas(env, smoothing_types, alphas, n_iterations=100, k_samples=40, q=0.5):

	for smoothing_type in smoothing_types:
		for alpha in alphas:
			print(f'model: alpha={alpha}, smoothing_type={smoothing_type}')
			agent = CrossEntropyAgent(state_n=500, action_n=6, smoothing=smoothing_type, alpha=alpha)
			agent, average_rewards = teach_model(agent, env, n_iterations, k_samples, q)

			plt.plot(np.arange(len(average_rewards)), average_rewards, label=f'alpha={alpha}')
			plt.legend()
		plt.savefig(f'(with_alpha_decay smoothing={smoothing_type}, n={n_iterations},k={k_samples}, q={q} and different alphas.png')
		plt.clf()


#НАЧАЛО ЗДЕСЬ
env = gym.make('Taxi-v3')

smoothing_types = ['policy']
alphas = [0.1]
find_optimal_alphas(env, smoothing_types, alphas, k_samples=50)


#считаем среднее и дисперсию наград после 3000 итераций - берем 2000 значений
rewards_after = all_rewards[4000:]
mean = np.mean(rewards_after)
std = np.std(rewards_after)

plt.plot(all_rewards, label=f'alpha={alphas[0]}, k={50}\nmean={mean}, std={std}')
plt.legend()
plt.savefig(f'(number of trajectories X total_rewards.png')
plt.clf()