
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


# функция обучения модели; вынесена в отдельную функцию для удобства
def teach_model(agent, env, n_iterations, k_samples, q):
	average_rewards = []
	# обучение
	for i in range(n_iterations):
		trajectories = [get_trajectory(agent, env) for _ in range(k_samples)]

		# оценка модели
		total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
		average_rewards.append(np.mean(total_rewards))
		print(f'mean reward on {i} iteration: {np.mean(total_rewards)}')
		#print(f'min reward on {i} iteration: {np.min(total_rewards)}\n')

		elite_trajectories = []
		quantile = np.quantile(total_rewards, q)
		for trajectory in trajectories:
			if np.sum(trajectory['rewards']) > quantile:
				elite_trajectories.append(trajectory)

		# улучшение модели
		agent.fit(elite_trajectories)

	return agent, average_rewards



# функция для поиска оптимальных гиперпараметров K и q
# производится полный перебор всех возможных пар
def find_optimal_hyperparameters(env, q_list, k_samples=50, n_iterations=100):
		print(f'find optimal params with k={k_samples} and n={n_iterations}')
		for q in q_list:
			mean_rewards = []
			# усредняем значения для двух моделей, чтобы получить более гладкий график
			for i in range(2):
				agent = CrossEntropyAgent(state_n=500, action_n=6)
				agent, average_rewards = teach_model(agent, env, n_iterations, k_samples, q)
				counter = 0
				for state in agent.model:
					if float(1) in state:
						counter += 1
				print(counter)
				mean_rewards.append(average_rewards)
			mean_rewards = np.array(mean_rewards)
			mean_rewards = np.mean(mean_rewards, axis=0)
			plt.plot(np.arange(len(mean_rewards)), mean_rewards, label=f'q={q}')
			plt.legend()
		plt.savefig(f'n={n_iterations},k={k_samples} and different q.png')
		plt.clf()


#НАЧАЛО ЗДЕСЬ
env = gym.make('Taxi-v3')

# гиперпараметр q: задает количество(долю) элитных траекторий
q_list = [0.5]
# гиперпараметр K: задает количество сэмплированных траекторий
k_samples_list = [20]
for k_samples in k_samples_list:
	find_optimal_hyperparameters(env, q_list, k_samples=k_samples)