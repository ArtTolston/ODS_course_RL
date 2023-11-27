import numpy as np

class CrossEntropyAgent:
		def __init__(self, state_n=25, action_n=4):
			self.state_n = state_n
			self.action_n = action_n
			self.model = np.ones((self.state_n, self.action_n), dtype=float) / self.action_n

		def get_action(self, state):
			action = np.random.choice(self.action_n, p=self.model[state])
			return action

		def get_state(self, obs):
			return int(np.sqrt(self.state_n) * obs[0] + obs[1])

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

