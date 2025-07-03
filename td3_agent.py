import copy
import random
from collections import namedtuple, deque
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import os
from base_agent import BaseAgent

# GRU processing type: 'fusion' combines F-measure and action features, 'full' uses raw GRU output
gru_type = 'fusion'

# Transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Actor Network: maps state to action
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()
        self.action_range = action_range
        self.hidden_size = 128

        self.gru = nn.GRU(5, self.hidden_size, batch_first=True)
        self.gru_fmeasure = nn.GRU(3, self.hidden_size, batch_first=True)
        self.gru_action = nn.GRU(2, self.hidden_size, batch_first=True)

        self.gru_linear_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU()
        )

        self.linear_active_stack = nn.Sequential(
            nn.Linear(self.hidden_size, action_dim),
            nn.Tanh()  # Output in (-1, 1)
        )

    def forward(self, state):
        if gru_type == 'full':
            shared_features = self.gru(state.reshape(len(state), -1, 5))[1][-1]
        elif gru_type == 'fusion':
            state = state.reshape(len(state), -1, 5)
            f_out = self.gru_fmeasure(state[:, :, :3])[1][-1]
            a_out = self.gru_action(state[:, :, 3:])[1][-1]
            shared_features = self.gru_linear_fusion(torch.cat([f_out, a_out], dim=1))
        else:
            shared_features = state
        return self.linear_active_stack(shared_features)

# Critic Network: estimates Q-value for (state, action) pairs
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden_size = 128

        self.gru = nn.GRU(5, self.hidden_size, batch_first=True)
        self.gru_fmeasure = nn.GRU(3, self.hidden_size, batch_first=True)
        self.gru_action = nn.GRU(2, self.hidden_size, batch_first=True)

        self.gru_linear_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU()
        )

        self.linear_active_stack = nn.Sequential(
            nn.Linear(self.hidden_size + action_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, state, action):
        if gru_type == 'full':
            shared_features = self.gru(state.reshape(len(state), -1, 5))[1][-1]
        elif gru_type == 'fusion':
            state = state.reshape(len(state), -1, 5)
            f_out = self.gru_fmeasure(state[:, :, :3])[1][-1]
            a_out = self.gru_action(state[:, :, 3:])[1][-1]

            shared_features = self.gru_linear_fusion(torch.cat([f_out, a_out], dim=1))
        else:
            shared_features = state

        return self.linear_active_stack(torch.cat([shared_features, action], dim=1))

# TD3 Agent definition
class TD3Agent(BaseAgent):
    def __init__(self, state_dim, action_dim, learning_rate, env, device, start_time,
                 replay_buffer_size, batch_size, total_episodes):
        super().__init__(env, device, state_dim, action_dim)

        # Action range setup (normalized to [-1, 1])
        self.action_range = torch.tensor([[-1, 1], [-1, 1]], device=device)
        self.env_action_range = torch.tensor(env.action_space, device=device)

        # Initialize actor and critic networks and their targets
        self.actor_network = ActorNetwork(state_dim, action_dim, self.action_range).to(device)
        self.target_actor_network = copy.deepcopy(self.actor_network)
        self.actor_optimizer = optim.AdamW(self.actor_network.parameters(), lr=learning_rate, amsgrad=True)

        self.critic_network_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic_network_1 = copy.deepcopy(self.critic_network_1)
        self.critic_optimizer_1 = optim.AdamW(self.critic_network_1.parameters(), lr=learning_rate, amsgrad=True)

        self.critic_network_2 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic_network_2 = copy.deepcopy(self.critic_network_2)
        self.critic_optimizer_2 = optim.AdamW(self.critic_network_2.parameters(), lr=learning_rate, amsgrad=True)

        # Replay buffer and training parameters
        self.replay_memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target network update rate
        self.total_episodes = total_episodes
        self.max_step = 10000
        self.start_time = start_time

        # Noise settings for exploration and policy smoothing
        self.exploration_noise_std = 0.1
        self.policy_noise_std = 0.2
        self.noise_clip = 0.4

        self.optimize_count = 0
        self.delay_update_frequency = 5

        # Best model tracking: [f_measure, drift_count, episode, model_dict]
        self.best_model = [0, 1000, 0, None]

        # Metrics
        self.episode_durations = []

    # actor_network feedforward
    def make_action(self, state):
        action = self.actor_network(torch.tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0))
        return action.squeeze(0).cpu().numpy()

    # actor_network sampling and selection
    def sample_action(self, state):
        raw_action = self.actor_network(state)
        noise = torch.randn(self.action_dim, device=self.device)
        exploration_noise = noise * self.exploration_noise_std * ((self.action_range[:,1]-self.action_range[:,0]) * 0.5)
        action = (raw_action + exploration_noise).clamp(self.action_range[:,0], self.action_range[:,1])
        return action
    
    # action scale
    def scale_action(self, action, action_range):
        action_scaled = action_range[:, 0] + (action + 1) * 0.5 * (action_range[:, 1] - action_range[:, 0])
        return action_scaled

    
    def train(self, model_dir):
        self.actor_network.train()
        self.critic_network_1.train()
        self.critic_network_2.train()
        total_rewards,average_rewards, final_rewards = [], [], []  
        episode_actor_loss,episode_critic_loss=[],[]
        episode_f_measure, episode_drift_count = [], []

        prg_bar = tqdm(range(self.total_episodes))
        for i_episode in prg_bar:
            total_reward = 0
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0)  
            self.current_episode_actor_loss = []
            self.current_episode_critic_loss = []
            action_1_list, action_2_list = [], []
            memory_size_list, sampling_rate_list = [], []
            for t in tqdm(range(self.max_step)):
                with torch.no_grad():
                    action = self.sample_action(state)
                observation, reward, done, info = self.env.step(self.scale_action(action, self.env_action_range).squeeze(0).cpu().numpy())
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                actor_output = action.squeeze(0).cpu().numpy()
                action_1_list.append(actor_output[0])
                action_2_list.append(actor_output[1])
                memory_size_list.append(self.env.memory_size)
                sampling_rate_list.append(self.env.sampling_rate)
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.replay_memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()

                # if terminated or truncated:
                if done or t==self.max_step-1:
                    self.episode_durations.append(t + 1)
                    final_rewards.append(reward.item())
                    total_rewards.append(total_reward)
                    average_rewards.append(total_reward/(t + 1))
                    episode_f_measure.append(np.array(self.env.evaluations[1:])[:,2].mean())
                    episode_drift_count.append(len(self.env.drift_moments)-1)
                    episode_actor_loss.append(sum(self.current_episode_actor_loss)/len(self.current_episode_actor_loss) if len(self.current_episode_actor_loss)>0 else 0)
                    episode_critic_loss.append(sum(self.current_episode_critic_loss)/len(self.current_episode_critic_loss) if len(self.current_episode_critic_loss)>0 else 0)
                    break
            
            print(f"TD3, Episode: {i_episode}, actor_loss: {episode_actor_loss[-1]: 4.3f}, critic_loss: {episode_critic_loss[-1]: 4.3f}, Total_Reward: {total_rewards[-1]: 4.3f}, Average_Reward: {average_rewards[-1]: 4.3f}, Final_Reward: {final_rewards[-1]: 4.3f}, episode_duration: {self.episode_durations[-1]}, optimize_count: {self.optimize_count}")

            test_f_measure, test_drift_count = self.agent_test()
            if test_f_measure > self.best_model[0]:
                self.best_model[0] = test_f_measure
                self.best_model[1] = test_drift_count
                self.best_model[2] = i_episode+1
                self.best_model[3] = {
                                        "actor_network": copy.deepcopy(self.actor_network.state_dict()),
                                        "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
                                        "critic_network_1": copy.deepcopy(self.critic_network_1.state_dict()),
                                        "critic_optimizer_1": copy.deepcopy(self.critic_optimizer_1.state_dict()),
                                        "critic_network_2": copy.deepcopy(self.critic_network_2.state_dict()),
                                        "critic_optimizer_2": copy.deepcopy(self.critic_optimizer_2.state_dict())
                                    }

            if (i_episode+1)%50 == 0:
                os.makedirs(os.path.join(model_dir, self.start_time), exist_ok=True)
                self.save(os.path.join(model_dir, self.start_time, 'td3_agent_model_episode_'+ str(i_episode+1) +'.pth'))
                plt.figure()
                plt.subplot(2, 3, 1)
                plt.plot(total_rewards)
                plt.title("Total Rewards")
                plt.subplot(2, 3, 2)
                plt.plot(average_rewards)
                plt.title("Average Rewards")
                plt.subplot(2, 3, 3)
                plt.plot(episode_f_measure)
                plt.title("f_measure")
                plt.subplot(2, 3, 4)
                plt.plot(episode_actor_loss)
                plt.title("actor loss")
                plt.subplot(2, 3, 5)
                plt.plot(episode_critic_loss)
                plt.title("critic loss")
                plt.subplot(2, 3, 6)
                plt.plot(episode_drift_count)
                plt.title("drift_count")
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, self.start_time, 'td3_performance_episode_'+ str(i_episode+1) +'.png'))
                plt.close()
        os.makedirs(os.path.join(model_dir, self.start_time), exist_ok=True)
        torch.save(self.best_model[3], os.path.join(model_dir, self.start_time, 'td3_agent_best_model.pth'))
        print('save best model_episode_'+ str(self.best_model[2]),'f_measure:',self.best_model[0],'drift_count:',self.best_model[1])

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.optimize_count += 1
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        state_action_values_1 = self.critic_network_1(state_batch, action_batch)
        state_action_values_2 = self.critic_network_2(state_batch, action_batch)
        
        next_state_values_1 = torch.zeros_like(state_action_values_1, device=self.device)
        next_state_values_2 = torch.zeros_like(state_action_values_2, device=self.device)
        with torch.no_grad():
            next_action_batch = self.target_actor_network(non_final_next_states)
            policy_noise = (torch.randn_like(next_action_batch) * self.policy_noise_std).clamp(-self.noise_clip, self.noise_clip)
            smoothed_next_action_batch = (next_action_batch + policy_noise * ((self.action_range[:,1]-self.action_range[:,0]) * 0.5)).clamp(self.action_range[:,0], self.action_range[:,1])
            next_state_values_1[non_final_mask] = self.target_critic_network_1(non_final_next_states, smoothed_next_action_batch)
            next_state_values_2[non_final_mask] = self.target_critic_network_2(non_final_next_states, smoothed_next_action_batch)

            target_value = torch.min(next_state_values_1,next_state_values_2)
            expected_state_action_values = target_value * self.gamma + reward_batch.unsqueeze(1)

        # Compute critic loss
        critic_loss_1 = nn.functional.mse_loss(state_action_values_1, expected_state_action_values)
        critic_loss_2 = nn.functional.mse_loss(state_action_values_2, expected_state_action_values)
        self.current_episode_critic_loss.append(critic_loss_1.item()+critic_loss_2.item())
        # Optimize the critic
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if(self.optimize_count % self.delay_update_frequency == 0):
            # compute actor loss
            actor_loss = - self.critic_network_1(state_batch,self.actor_network(state_batch)).mean()
            self.current_episode_actor_loss.append(actor_loss.item())
            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor_network.parameters(), self.target_actor_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_network_1.parameters(), self.target_critic_network_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_network_2.parameters(), self.target_critic_network_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def test(self):
        self.actor_network.eval()
        self.critic_network_1.eval()
        self.critic_network_2.eval()
        with torch.no_grad():
            super().test()
    
    def agent_test(self):
        self.actor_network.eval()
        self.env.multi_reward = False
        self.env.multi_log = False
        state, info = self.env.reset()
        episode_reward = []
        action_1_list, action_2_list = [], []
        memory_size_list, sampling_rate_list = [], []
        while True:
            with torch.no_grad():
                action = self.make_action(state)
            state, reward, done, info = self.env.step(np.squeeze(self.scale_action(np.expand_dims(action,0),np.array(self.env.action_space)),0))
            action_1_list.append(action[0])
            action_2_list.append(action[1])
            memory_size_list.append(self.env.memory_size)
            sampling_rate_list.append(self.env.sampling_rate)
            episode_reward.append(reward) 
            if done:
                print("test log: ", self.env.log_name,np.array(self.env.evaluations[1:])[:,2].mean(), len(self.env.drift_moments)-1)
                break
        self.env.multi_reward = True
        self.env.multi_log = True
      
        self.actor_network.train()
        return np.array(self.env.evaluations[1:])[:,2].mean(), len(self.env.drift_moments)-1

    def save(self, model_path):
        agent_dict = {
            "actor_network": self.actor_network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_network_1": self.critic_network_1.state_dict(),
            "critic_optimizer_1": self.critic_optimizer_1.state_dict(),
            "critic_network_2": self.critic_network_2.state_dict(),
            "critic_optimizer_2": self.critic_optimizer_2.state_dict()
        }
        torch.save(agent_dict, model_path)
        print('saving model: ', model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_network_1.load_state_dict(checkpoint["critic_network_1"])
        self.critic_optimizer_1.load_state_dict(checkpoint["critic_optimizer_1"])
        self.critic_network_2.load_state_dict(checkpoint["critic_network_2"])
        self.critic_optimizer_2.load_state_dict(checkpoint["critic_optimizer_2"])
        print('loading model: ', model_path)
