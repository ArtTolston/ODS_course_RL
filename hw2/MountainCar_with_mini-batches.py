import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn 

import numpy as np

import gym

import matplotlib.pyplot as plt

import time

from reward_shaping import reward_shaping, potential, reward_shaping_new


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

        self.layers_list = [self.network[0], self.network[2]]

        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()

        self.epoch = 0
        self.init_eps = 20.0
        self.eps = self.init_eps
        self.lambda_eps = lambda epoch: 0.96 ** epoch

        self.gradient_norms = []
        self.losses = []

        self.batch_size = 32
        
    def forward(self, _input):
        return self.network(_input) 
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        determ_action = self.forward(state)
        action = self.tanh(determ_action + torch.tensor(np.random.uniform(low=-self.eps / 2, high=self.eps / 2, size=1))).detach().numpy()
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
        elite_actions = torch.FloatTensor(elite_actions)

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

        #средние значения по нормам градиентов и лоссам
        self.gradient_norms.append(np.mean(grad_norm))
        self.losses.append(np.mean(iter_loss))

        self.epoch += 1
        self.eps = self.init_eps * self.lambda_eps(self.epoch)
        
        
def get_trajectory(env, agent, trajectory_len = 1000, visualize=False, reward_sh='base'):
    trajectory = {'states':[], 'actions': [], 'total_reward': []}
    
    state = env.reset()
    trajectory['states'].append(state)

    max_reward = -1.0
    
    for i in range(trajectory_len):
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        next_state, reward, done, _ = env.step(action)
        # здесь нужен reward shaping

        if reward_sh == 'base':
            gamma = 1.0
            reward = reward_shaping(state, next_state, reward, potential, gamma=gamma)
            trajectory['total_reward'].append(gamma * reward)
        elif reward_sh == 'velocity':
            reward = reward_shaping_new(state, next_state, action, reward)
            trajectory['total_reward'].append(reward)
        else:
            trajectory['total_reward'].append(reward)
    
        if visualize:
            print(f'state: {state}, action: {action}, reward: {np.round(reward, 3)}')
            env.render()

        state = next_state

        if done:
            break
            
        trajectory['states'].append(state)

    # trajectory['total_reward'] += max_reward
            
    return trajectory

def get_elite_trajectories(trajectories, q_param):
    total_rewards = [np.sum(trajectory['total_reward']) for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param) 
    return [trajectory for trajectory in trajectories if np.sum(trajectory['total_reward']) > quantile]


def teach_model(env, agent, episode_n, trajectory_n, q_param):  
    average_rewards = []
    training_time_start = time.time()
    for episode in range(episode_n):
        start = time.time()
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        
        mean_total_reward = np.mean([np.sum(trajectory['total_reward']) for trajectory in trajectories])
        average_rewards.append(mean_total_reward)
        elite_trajectories = get_elite_trajectories(trajectories, q_param)
        print(f'#### num elite: {len(elite_trajectories)}, num successed: {len([trajectory for trajectory in elite_trajectories if trajectory["total_reward"][-1] > 95])}')
        
        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories)
        print(f'episode: {episode}, mean_total_reward = {mean_total_reward}  ### time: {time.time() - start}')
    print(f'##### total training time: {time.time() - training_time_start}')

    return agent, average_rewards 

  
def save_checkpoint(model, optimizer, checkpoint_path='./checkpoint_clear_MountainCar.pth'):
    state = {'model_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict()}
    torch.save(state, checkpoint_path) 


def grid_search_k_v(env, episode_n, k_list, q_list):
    plt.figure(figsize=(45, 15))
    for k in k_list:
        for q in q_list:
            print(f'learn simple model with parameters: k={k}, q={q}')
            agent = CEM(2, 1)
            agent, rewards =teach_model(env, agent, episode_n, k, q)
            # if q == 0.95 and k == 60:
            #     save_checkpoint(agent, agent.optimizer)
            plt.subplot(3, 1, 1)
            plt.plot(rewards, label=f'q={q}, max={np.max(rewards)}')
            plt.legend()
            #get_trajectory(env, agent, visualize=True)
            plt.subplot(3, 1, 2)
            plt.plot(agent.losses, label=f'q={q}, max={np.max(agent.losses)}, min={np.min(agent.losses)}')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(agent.gradient_norms, label=f'q={q}, max={np.max(agent.gradient_norms)}, min={np.min(agent.gradient_norms)}')
            plt.legend()
        plt.savefig(f'base MountainCar with mini-batches k={k}.png')
        plt.clf()


env = gym.make('MountainCarContinuous-v0')
state_dim = 2
action_n = 1

agent = CEM(state_dim, action_n)
episode_n = 100

k_list = [60]
q_list = [0.95]

grid_search_k_v(env, episode_n, k_list, q_list)

# get_trajectory(env, agent, trajectory_len, visualize=True)


