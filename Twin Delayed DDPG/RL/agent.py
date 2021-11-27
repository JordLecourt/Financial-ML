import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Actor, Critic
import base64
from io import BytesIO
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    """
    Agent class that handles the training of the networks and provides outputs as actions
    
    :param state_dim (int): state size
    :param action_dim (int): action size
    :param max_action (float): highest action to take
    :param device (device): cuda or cpu to process tensors
    :param env (env): gym environment to use
    """
    def __init__(self, algorithm, symbol, state_dim, action_dim, max_action, seed, h1_units=400, h2_units=300, warmup=1000):
        self.is_trained = False
        self.algorithm = algorithm
        self.symbol = symbol
        
        self.actor = Actor(state_dim, action_dim, max_action, seed, h1_units, h2_units).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, seed, h1_units, h2_units).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim, seed, h1_units, h2_units).to(device)
        self.critic_target = Critic(state_dim, action_dim, seed, h1_units, h2_units).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.warmup = warmup
        self.time_step = 0
        

    def select_action(self, state, noise=0.1):
        """
        Select an appropriate action from the agent policy
        
        :param state (array): current state of environment
        :param noise (float): how much noise to add to acitons
        
        :return action (float): action clipped within action range
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=1))
        
        self.time_step += 1    
        return action.clip(-self.max_action, self.max_action)

    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        Train and update actor and critic networks
        
        :param replay_buffer (ReplayBuffer): buffer for experience replay
        :param iterations (int): how many times to run training
        :param batch_size(int): batch size to sample from replay buffer
        :param discount (float): discount factor
        :param tau (float): soft update for main networks to target networks
                
        :return actor_loss (float): loss from actor network
        :return critic_loss (float): loss from critic network
        """
        
        if replay_buffer.cntr < batch_size:
            return
        
        for it in range(iterations):

            # Sample replay buffer 
            _state, _next_state, _action, _reward, _done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(_state.reshape(batch_size, -1)).to(device)
            next_state = torch.FloatTensor(_next_state.reshape(batch_size, -1)).to(device)
            action = torch.FloatTensor(_action).to(device)
            reward = torch.FloatTensor(_reward).to(device)
            done = torch.FloatTensor(1 - _done).to(device)

            # Select action according to policy and add clipped noise
            next_action = self.actor_target(next_state)
            next_action = next_action + torch.clamp(torch.tensor(np.random.normal(scale=policy_noise)), -noise_clip, noise_clip)
            next_action = torch.clamp(next_action, -self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), '%s_%s_actor.pth' % (self.symbol, filename))
        torch.save(self.critic.state_dict(), '%s_%s_critic.pth' % (self.symbol, filename))

    def load(self, filename):
        self.actor.load_state_dict(torch.load('%s_%s_actor.pth' % (self.symbol, filename)))
        self.critic.load_state_dict(torch.load('%s_%s_critic.pth' % (self.symbol, filename)))