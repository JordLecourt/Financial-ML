from consolidator import VolumeBarConsolidator
from QuantConnect.Indicators import *

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from RL.agent import TD3
from RL.environment import TradingEnv
from RL.runner import Runner

class RLAlphaModel(AlphaModel):
    
    def __init__(self):
        self.symbols = {}
        self.Name = "RLAlphaModel"
        
    def Update(self, algorithm, data):
        insights = []
        return insights
         
    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbols:
                self.symbols[symbol] = SymbolData(algorithm, symbol)
                
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbols:
                symbolData = self.symbols.pop(symbol, None)
                if symbolData:
                    symbolData.RemoveConsolidator()
        
class SymbolData:
    
    def __init__(self, algorithm, symbol):
        self.algorithm = algorithm
        self.symbol = symbol
        self.raw_features = None        # The indicators have not been through the PCA
        self.signal_features = None     # The indicators have been through the PCA
        self.is_warmed_up = False       # Bool to indicate if the PCA has been fit
        
        """
        Add volume bar consolidator
        
        Lopez de Prado suggested that using 1/50 of the average daily 
        volume would result in more desirable statistical properties
        """
        # Get the average daily volume
        history = algorithm.History(symbol, 14, Resolution.Daily)
        volumeSMA = SimpleMovingAverage(14)
        for bar in history.itertuples():
            volumeSMA.Update(bar.Index[1], bar.volume)
            
        self.barSize = round(volumeSMA.Current.Value / 50)
        
        self.consolidator = VolumeBarConsolidator(self.barSize)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        
        algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        
        """
        Indicators passed to the agent
        """
        self._macd = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.Exponential)
        self._rsi = RelativeStrengthIndex(14, MovingAverageType.Simple)
        self._adx = AverageDirectionalIndex(14)
        self._cci = CommodityChannelIndex(14, MovingAverageType.Simple)
        self._obv = OnBalanceVolume()
        
        """
        Warm up the indicators
        """
        history = algorithm.History(self.symbol, timedelta(days=10), Resolution.Second)
        for idx, bar in history.iterrows():
            tradeBar = TradeBar(idx[1], symbol, bar.open, bar.high, bar.low, bar.close, bar.volume)
            self.consolidator.Update(tradeBar)
            self.consolidator.Scan(idx[1])
            
        """
        Fit a PCA to the indicators
        """
        self.pca = PCA(n_components=3, svd_solver='auto')
        processed_columns = self.pca.fit_transform(self.raw_features.iloc[:, 5:])
        processed_columns = pd.DataFrame(processed_columns, columns=['pca1', 'pca2', 'pca3'])
        self.signal_features = pd.concat([ self.raw_features.iloc[:, :5], processed_columns ], axis=1)
        self.is_warmed_up = True
        
        """
        Train the agent on historical data for the new symbol
        """
        window_size = 10
        train_end_tick = round(self.length() * 0.8)
        trainEnv = TradingEnv(symbol_data=self, window_size=window_size, start_tick=0, end_tick=train_end_tick)
        testEnv = TradingEnv(symbol_data=self, window_size=window_size, start_tick=train_end_tick, end_tick=None)
        
        state_dim = self.width() * window_size
        action_dim = trainEnv.action_space.shape[0]
        max_action = 1
        seed = 0
        self.agent = TD3(algorithm, self.symbol, state_dim, action_dim, max_action, \
                        seed, h1_units=400, h2_units=300, warmup=1000)
                        
        runner = Runner(algorithm, agent=self.agent, n_episodes=100, batch_size=32, gamma=0.99, \
                    tau=0.005, noise=0.2, noise_clip=0.5, explore_noise=0.1, policy_frequency=2)
        runner.train(trainEnv, testEnv)
        
    def RemoveConsolidator(self):
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, IDataConsolidator(self.consolidator))
        
    def OnDataConsolidated(self, sender, bar):
        self._macd.Update(bar.EndTime, bar.Close)
        self._rsi.Update(bar.EndTime, bar.Close)
        self._adx.Update(bar)
        self._cci.Update(bar)
        self._obv.Update(bar)
        self._process_data(bar)
        
    def _process_data(self, bar):
        """ The features passed to the agent are selected in this function. To add one
            simply add it to the dataframe.
            
            Currently, the feature are:
                open, hight, low, close
                MACD, RSI, ADX, CCI, OBV (passed through a PCA to remove correlation)
        """
        if not self._macd.IsReady or \
            not self._rsi.IsReady or \
            not self._adx.IsReady or \
            not self._cci.IsReady or \
            not self._obv.IsReady:
                return
        
        if not self.is_warmed_up:
            columns = [ 'endTime', 'open', 'hight', 'low', 'close', 'macd', 'rsi', 'adx', 'cci', 'obv' ]
            self.raw_features = pd.concat([self.raw_features, 
                                            pd.DataFrame([[ bar.EndTime,
                                                            bar.Open, 
                                                            bar.High, 
                                                            bar.Low, 
                                                            bar.Close,
                                                            self._macd.Current.Value,
                                                            self._rsi.Current.Value,
                                                            self._adx.Current.Value,
                                                            self._cci.Current.Value,
                                                            self._obv.Current.Value   ]], 
                                                            columns=columns)],
                                            ignore_index=True)
        else:
            columns = [ 'endTime', 'open', 'hight', 'low', 'close', 'pca1', 'pca2', 'pca3' ]
            raw_feature = [[    self._macd.Current.Value,
                                self._rsi.Current.Value,
                                self._adx.Current.Value,
                                self._cci.Current.Value,
                                self._obv.Current.Value     ]]
            signal_feature = self.pca.transform(raw_feature)
            
            self.signal_features = pd.concat([self.signal_features, 
                                              pd.DataFrame([[   bar.EndTime,
                                                                bar.Open, 
                                                                bar.High, 
                                                                bar.Low, 
                                                                bar.Close,
                                                                signal_feature[0][0],
                                                                signal_feature[0][1],
                                                                signal_feature[0][2] ]], 
                                                                columns=columns)],
                                            ignore_index=True)
    
    def length(self):
        return self.signal_features.shape[0] if self.signal_features is not None else 0
        
    def width(self):
        return self.signal_features.shape[1] - 1 if self.signal_features is not None else 0
    
    def get_observation(self, window_size=10, current_tick=None): 
        if current_tick is None:
            return self.signal_features.tail(window_size).iloc[:, 1:]
        else:
            return self.signal_features[(current_tick - window_size): current_tick].iloc[:, 1:]