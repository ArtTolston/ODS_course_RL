import gym
import numpy as np
import time
import matplotlib.pyplot as plt


class CrossEntropyAgent:
		def __init__(self, state_n=25, action_n=4):
			self.state_n = state_n
			self.action_n = action_n
			self.model = np.ones((self.state_n, self.action_n), dtype=float) / self.action_n

		def get_action(self, state):
			action = np.random.choice(self.action_n, p=self.model[state])
			return action

		def fit(self, elite_trajectories):
			new_model = np.zeros((self.state_n, self.action_n), dtype=float)

			for trajectory in elite_trajectories:
				for state, action in zip(trajectory['states'], trajectory['actions']):
					new_model[state][action] += 1

			for state in range(self.state_n):
				if np.sum(new_model[state]) > 0:
					new_model[state] /= np.sum(new_model[state])
				else:
					new_model[state] = self.model[state].copy()

			self.model = new_model
			return None


# класс, в котором каждое действие в каждом состоянии является детерминированным
# model - вектор с длиной равной размеру пространства состояний 
class DeterministicAgent:
	def __init__(self, model):
		self.model = model

	def get_action(self, state):
		action = self.model[state]
		return action

# функция для сэмплирования k детерминированных агентов из одного стохастического
def make_deterministic_from_stochastic(agent, k):
	deterministic_agents = []
	for _ in range(k):
		model = []
		for state in range(agent.state_n):
			model.append(agent.get_action(state))
		deterministic_agents.append(DeterministicAgent(model))
	return deterministic_agents


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



def teach_model(agent, env, n_iterations, k_samples, q, m=4):
	average_rewards = []
	#обучение
	for i in range(n_iterations):
		total_rewards = np.array([])
		trajectories = np.array([])
		# добавляется еще один цикл по каждому детерминированному агенту
		for determ_agent in make_deterministic_from_stochastic(agent, m):
			determ_trajectories = [get_trajectory(determ_agent, env) for _ in range(k_samples)]

			#оценка модели
			total_rewards = np.concatenate((total_rewards, [np.sum(trajectory['rewards']) for trajectory in determ_trajectories]))
			trajectories = np.concatenate((trajectories, determ_trajectories))
		
		average_rewards.append(np.mean(total_rewards))	
		print(f'mean reward on {i} iteration: {np.mean(total_rewards)}')	

		elite_trajectories = []
		quantile = np.quantile(total_rewards, q)
		for trajectory in trajectories:
			if np.sum(trajectory['rewards']) > quantile:
				elite_trajectories.append(trajectory)

			#улучшение модели
			agent.fit(elite_trajectories)

	return agent, average_rewards



# анализируем как обучаемость зависит от М
def find_optimal_M(env, M_list, n_iterations=10, k_samples=40, q=0.5):
	for M in M_list:
		agent = CrossEntropyAgent(state_n=500, action_n=6)
		agent, average_rewards = teach_model(agent, env, n_iterations, k_samples, q, m=M)
		plt.plot(np.arange(len(average_rewards)), average_rewards, label=f'M={M}')
		plt.legend()
	plt.savefig(f'NEWstochastic enviroment: n={n_iterations}, k={k_samples}, q={q} and different M.png')
	plt.clf()


#НАЧАЛО ЗДЕСЬ
env = gym.make('Taxi-v3')



# q_list = [0.5, 0.7, 0.8]
# k_samples_list = [20, 40, 60]
# for k_samples in k_samples_list:
# 	find_optimal_hyperparameters(env, q_list, k_samples=k_samples)

#гиперпараметр N: задает количество итераций
n_iterations = 10
#гиперпараметр K: задает количество сэмплированных траекторий
k_samples = 40
#гиперпараметр q: задает количество отобранных элитных траекторий
q = 0.5

M_list = [35]
find_optimal_M(env, M_list)

# M = 5
# agent = CrossEntropyAgent(state_n=500, action_n=6)
# agent, average_rewards = teach_model(agent, env, n_iterations, k_samples, q, m=M)


# plt.plot(np.arange(len(average_rewards)), average_rewards)
# plt.show()

# trajectory = get_trajectory(agent, env, visualize=True)
# print(f'total reward: {np.sum(trajectory["rewards"])}')