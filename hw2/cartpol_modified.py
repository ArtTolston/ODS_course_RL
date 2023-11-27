
import torch
from torch import nn 
import numpy as np
import gym
import matplotlib.pyplot as plt


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100), 
            nn.ReLU(), 
            nn.Linear(100, self.action_n)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.lambdaLR = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lambdaLR)
        self.loss = nn.CrossEntropyLoss()

        self.uniform_dist = np.ones(self.action_n, dtype=np.float) / self.action_n
        self.eps = 0.01
        
    def forward(self, _input):
        return self.network(_input) 
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        state = state[np.newaxis, :]
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()[0]
        actual_prob = (1.0 - self.eps) * action_prob + self.eps * self.uniform_dist
        print(action_prob)
        print(sum(actual_prob))
        action = np.random.choice(self.action_n, p=actual_prob)
        return action
    
    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        #print(elite_states)
        #print(elite_actions)
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        pred_actions = self.forward(elite_states)
        pred_actions = self.softmax(pred_actions)
        print(pred_actions[0])

        loss = self.loss(pred_actions, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        
def get_trajectory(env, agent, trajectory_len, visualize=False):
    trajectory = {'states':[], 'actions': [], 'total_reward': 0}
    
    state = env.reset()
    trajectory['states'].append(state)
    
    for _ in range(trajectory_len):
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward
        
        if done:
            break
            
        if visualize:
            env.render()
            
        trajectory['states'].append(state)
            
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]


env = gym.make('CartPole-v1')
state_dim = 4
action_n = 2

agent = CEM(state_dim, action_n)
episode_n = 200
trajectory_n = 20
trajectory_len = 500
q_param = 0.8

for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
        
get_trajectory(env, agent, trajectory_len, visualize=True)