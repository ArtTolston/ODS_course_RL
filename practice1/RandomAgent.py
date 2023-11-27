import numpy as np


class RandomAgent:
	def __init__(self):
		self.state_n = 25
		self.action_n = 4

	def get_action(self, state):
		action = np.random.randint(self.action_n)
		return action

	def get_state(self, obs):
		return int(np.sqrt(self.state_n) * obs[0] + obs[1])

	def get_trajectory(self, agent, enviroment):
		trajectory = {'states': [], 'actions': [], 'reward': []}

		
