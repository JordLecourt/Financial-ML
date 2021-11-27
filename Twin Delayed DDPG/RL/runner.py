from collections import deque
import pandas as pd
import numpy as np
import gym
import sys

from buffer import ReplayBuffer

class Runner:
    """Carries out the environment steps and adds experiences to memory"""
    
    def __init__(self, algorithm, agent=None, n_episodes=100, batch_size=32, gamma=0.99, \
                    tau=0.005, noise=0.2, noise_clip=0.5, explore_noise=0.1, policy_frequency=2):

        self.agent = agent
        self.algorithm = algorithm
        
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        self.noise = noise
        self.noise_clip = noise_clip
        self.explore_noise = explore_noise
        
        self.policy_frequency = policy_frequency
        self.replay_buffer = ReplayBuffer(algorithm)
        

    def evaluate_policy(self, testEnv, eval_episodes=10):
        avg_profit = 0.
        
        for i in range(eval_episodes):
            obs = testEnv.reset()
            done = False
            while not done:
                action = self.agent.select_action(np.array(obs), noise=0)
                obs, reward, done, _ = testEnv.step(action)
                
            avg_profit += testEnv.history['total_profit'][-1]

        avg_profit /= eval_episodes
        
        return avg_profit
    
        
    def train(self, trainEnv, testEnv):
        best_profit = -1
        
        for i_episode in range(1, self.n_episodes + 1):
            
            obs = trainEnv.reset()
            done = False
            episode_timesteps = 0
            
            while not done:
                
                action = self.agent.select_action(np.array(obs), noise=self.explore_noise)
                new_obs, reward, done, _ = trainEnv.step(action)
                self.replay_buffer.add((obs, new_obs, action, reward, done))
                obs = new_obs
                
                episode_timesteps += 1
                
                self.agent.train(self.replay_buffer, 20, 
                                    self.batch_size, self.gamma, self.tau, 
                                    self.noise, self.noise_clip, self.policy_frequency)
                    
            profit = self.evaluate_policy(testEnv)
            
            if profit > best_profit:
                best_profit = profit
                self.algorithm.Debug(str(i_episode)+"| Best Model! |"+str(round(best_profit,3)))
                self.agent.save("best_avg")
                
        self.agent.is_trained = True