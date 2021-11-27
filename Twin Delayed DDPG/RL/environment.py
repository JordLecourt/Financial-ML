import gym
from gym import spaces

import numpy as np
import pandas as pd
import random
from enum import Enum

from scipy.stats import linregress

class Actions(Enum):
    """ enum for the two actions the agent can take

        None = 0
        Sell = 1
        Buy = 2
    """
    Null = 0
    Sell = 0
    Buy = 1
    
class Positions(Enum):
    """ enum for the two positions the agent can be in
        
        None = 0
        Short = 1
        Long = 2
    """    
    Null = 0
    Short = 0
    Long = 1

class TradingEnv(gym.Env):
    
    def __init__(self,  symbol_data=None, window_size=10, start_tick=0, end_tick=None):
        
        self.window_size = window_size
        self.symbol_data = symbol_data
        self.isReady = False
        
        self.action_space = spaces.Box(-1, +1, (1,), dtype=np.float32)
        
        # transaction fee (in %)
        self.trade_fee_bid_percent = 0.005
        self.trade_fee_ask_percent = 0.005
        
        # episode
        self._start_tick = start_tick + self.window_size
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self.history = None
        
        if end_tick is None:
            self._end_tick = symbol_data.length() - 1
        else:
            self._end_tick = end_tick - 1
    
        
    def reset(self, randomIndex=False):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = -1
        self._position = Positions.Null
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.
        self.history = {}
        
        observations = self._get_observation()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(observations),), dtype=np.float32)
        
        return observations
        
    
    def _get_observation(self):
        return self.symbol_data.get_observation(self.window_size, self._current_tick)
        
        
    def _is_closing_trade(self, action):
        if ((action == Actions.Buy and self._position == Positions.Short) or 
            (action == Actions.Sell and self._position == Positions.Long)):
            return True
        else:
            return False
            
        
    def _get_reward(self, action):
        step_reward = 0
        
        # If we are closing a position, get the profit/loss    
        if self._is_closing_trade(action):
            close_index = self.symbol_data.signal_features.columns.get_loc('close')
            current_price = self.symbol_data.signal_features.iat[self._current_tick, close_index]
            last_trade_price = self.symbol_data.signal_features.iat[self._last_trade_tick, close_index]
            price_diff = current_price - last_trade_price
            
            if self._position == Positions.Long:
                step_reward += price_diff
            else:
                step_reward -= price_diff
                
        return step_reward
        
        
    def _update_profit(self, action):
        # If we are closing a position, update the total profit    
        if self._is_closing_trade(action) or self._done:
            close_index = self.symbol_data.signal_features.columns.get_loc('close')
            current_price = self.symbol_data.signal_features.iat[self._current_tick, close_index]
            last_trade_price = self.symbol_data.signal_features.iat[self._last_trade_tick, close_index]
            
            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
            else:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / current_price 
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * last_trade_price
        
        
    def step(self, action):
        # Get discrete action from float
        action_float = float(action)
        if action_float >= 0.1:     action = Actions.Buy
        elif action_float <= -0.1:  action = Actions.Sell
        else:                       action = Actions.Null
        
        self._done = False
        self._current_tick += 1
        
        if self._current_tick == self._end_tick:
            self._done = True
        
        step_reward = self._get_reward(action)
        self._total_reward = step_reward
        
        self._update_profit(action)
        
        # If the action close a trade, set the position to null
        if self._is_closing_trade(action):
            self._position = Positions.Null
            self._last_trade_tick = -1
            
        # If the position is null and the action is not null, open a trade
        elif self._position == Positions.Null and action != Actions.Null:
            self._last_trade_tick = self._current_tick
            if action == Action.Buy:    self._position = Positions.Long
            else:                       self._position = Positions.Short
                
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)
        
        return observation, step_reward, self._done, info
        
        
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
            
        for key, value in info.items():
            self.history[key].append(value)