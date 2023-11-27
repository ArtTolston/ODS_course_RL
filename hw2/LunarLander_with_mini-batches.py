
import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset

import numpy as np
import gym
import matplotlib.pyplot as plt

import time


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.fc1 = nn.Linear(state_dim, 50)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(50, action_n)
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        self.network = nn.Sequential(
            self.fc1, 
            self.rl1, 
            self.fc2
        )

        self.layers_list = [self.network[0], self.network[2]]
        
        self.softmax = nn.Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 120, 135], gamma=0.5)
        self.loss = nn.CrossEntropyLoss()

        self.gradient_norms = []
        self.losses = []

        self.batch_size = 128
        
    def forward(self, _input):
        return self.network(_input) 
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
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

        train_dataset = list(zip(elite_states, elite_actions))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        grad_norm = []
        iter_loss = []
        for x_batch, y_batch in train_loader:
            y_batch_pred = self.forward(x_batch)

            loss = self.loss(y_batch_pred, y_batch)
            loss.backward()

            for layer in self.layers_list:
                grad_norm.append(layer.weight.grad.norm())

            iter_loss.append(loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()
        #средние значения по нормам градиентов и лоссам
        self.gradient_norms.append(np.mean(grad_norm))
        self.losses.append(np.mean(iter_loss))
        
        
def get_trajectory(env, agent, trajectory_len = 1000, visualize=False):
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


def teach_model(env, agent, episode_n, trajectory_n, q_param):  
    average_rewards = []
    training_time_start = time.time()
    for episode in range(episode_n):
        start = time.time()
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        
        mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
        average_rewards.append(mean_total_reward)
        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        
        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)
        print(f'episode: {episode}, mean_total_reward = {mean_total_reward}  ### time: {time.time() - start}')
    print(f'##### total training time: {time.time() - training_time_start}')

    return agent, average_rewards  

  


def grid_search_k_v(env, episode_n, k_list, q_list):
    plt.figure(figsize=(45, 15))
    for k in k_list:
        for q in q_list:
            print(f'learn simple model with parameters: k={k}, q={q}')
            agent = CEM(8, 4)
            agent, rewards =teach_model(env, agent, episode_n, k, q)
            plt.plot(rewards, label=f'q={q}, max={np.max(rewards)}')
            plt.legend()
        plt.savefig(f'simple LunarLander with mini-batches k={k}.png')
        plt.clf()


env = gym.make('LunarLander-v2')
state_dim = 8
action_n = 4

episode_n = 150

k_list = [60]
q_list = [0.7]

grid_search_k_v(env, episode_n, k_list, q_list )


