
import torch
from torch import nn 
import numpy as np
import gym
import matplotlib.pyplot as plt

import time


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100), 
            nn.ReLU(), 
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.action_n)
        )

        self.layers_list = [self.network[0], self.network[2], self.network[4]]
        
        self.softmax = nn.Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.CrossEntropyLoss()


        self.gradient_norms = []
        self.losses = []
        
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

        pred_actions = self.forward(elite_states)

        loss = self.loss(pred_actions, elite_actions)
        loss.backward()

        grad_norm = []
        for layer in self.layers_list:
            grad_norm.append(layer.weight.grad.norm())
        self.gradient_norms.append(np.mean(grad_norm))

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.losses.append(float(loss.data.numpy()))
        
        
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
    plt.figure(figsize=(80, 45))
    for k in k_list:
        for q in q_list:
            print(f'learn simple model with parameters: k={k}, q={q}')
            agent = CEM(8, 4)
            agent, rewards =teach_model(env, agent, episode_n, k, q)
            plt.subplot(3, 1, 1)
            plt.plot(rewards, label=f'q={q}, max={np.max(rewards)}')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(agent.losses, label=f'q={q}, max={np.max(agent.losses)}, min={np.min(agent.losses)}')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(agent.gradient_norms, label=f'q={q}, max={np.max(agent.gradient_norms)}, min={np.min(agent.gradient_norms)}')
            plt.legend()
        plt.savefig(f'complex (with grads) LunarLander k={k}.png')
        plt.clf()
    
    




#env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')
state_dim = 8
action_n = 4

agent = CEM(state_dim, action_n)
episode_n = 100
trajectory_n = 20
trajectory_len = 1000
q_param = 0.8

k_list = [20]
q_list = [0.5]

grid_search_k_v(env, episode_n, k_list, q_list )

#get_trajectory(env, agent, trajectory_len, visualize=True)


