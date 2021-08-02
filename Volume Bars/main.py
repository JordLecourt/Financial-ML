from datetime import timedelta
from QuantConnect.Data.UniverseSelection import * 
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel
from mlfinlab.data_structures import standard_data_structures


class DummyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 2, 1)
        self.SetCash(100000) 
        
        # Universe Selection
        self.SetUniverseSelection(LiquidValueUniverseSelectionModel())
        self.UniverseSettings.Resolution = Resolution.Tick
        
        # Alpha Model
        self.SetAlpha(DummyAlphaModel())
        
        # Portfolio construction
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Execution model
        self.SetExecution(ImmediateExecutionModel())

    def OnData(self, data):
        pass
           
class LiquidValueUniverseSelectionModel(FundamentalUniverseSelectionModel):
    def __init__(self):
        super().__init__(True, None)
        self.lastMonth = -1
        
    def SelectCoarse(self, algorithm, coarse):
        # Change the universe only once a month
        if self.lastMonth == algorithm.Time.month:
            return Universe.Unchanged
        self.lastMonth = algorithm.Time.month
        
        # Sort Symbol by Dollar Volume
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        sortedByPrice = [x for x in sortedByDollarVolume if (x.Price > 10)]
        
        # Return the fist X
        return [x.Symbol for x in sortedByPrice[:2]]
        
       
class DummyAlphaModel(AlphaModel):
    def __init__(self):
        self.symbols = {}
        
    def Update(self, algorithm, data):
         insights = []
         return insights
         
    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbols:
                history = algorithm.History(symbol, 14, Resolution.Daily)
                self.symbols[symbol] = SymbolData(algorithm, symbol, history)
                
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbols:
                symbolData = self.symbols.pop(symbol, None)
                if symbolData:
                    symbolData.RemoveConsolidator()
        
class SymbolData:
    
    def __init__(self, algorithm, symbol, history):
        self.algorithm = algorithm
        self.symbol = symbol
        
        # Get the average daily volume
        volumeSMA = SimpleMovingAverage(14)
        for bar in history.itertuples():
            volumeSMA.Update(bar.Index[1], bar.volume)
            
        # Lopez de Prado suggested that using 1/50 of the average daily 
        # volume would result in more desirable statistical properties
        # “The Volume Clock: Insights into the high frequency paradigm” by Lopez de Prado, et al
        self.barSize = round(volumeSMA.Current.Value / 50)
        
        self.consolidator = VolumeBarConsolidator(self.barSize)
        self.consolidator.DataConsolidated += self.OnDataConsolidated
        
        algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
        
    def RemoveConsolidator(self):
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, IDataConsolidator(self.consolidator))
        
    def OnDataConsolidated(self, sender, bar):
        self.algorithm.Debug(f"Data Consolidatoed for {self.symbol} at {bar.EndTime} with bar: {bar}")
        
class VolumeBarConsolidator(PythonConsolidator):

    def __init__(self, barSize):

        #IDataConsolidator required vars for all consolidators
        self.Consolidated = None        #Most recently consolidated piece of data.
        self.WorkingData = None         #Data being currently consolidated
        self.InputType = TradeBar       #The type consumed by this consolidator
        self.OutputType = TradeBar      #The type produced by this consolidator

        #Consolidator Variables
        self.barSize = barSize
    
    def Update(self, data):
        # If the tick bar is marked as suspicious, return
        if data.Suspicious:
            return
        
        #If we don't have bar yet, create one
        if self.WorkingData is None:
            self.WorkingData = TradeBar(data.Time,data.Symbol, data.LastPrice, data.LastPrice,
                                        data.LastPrice, data.LastPrice, data.Quantity)

        #Update bar using TradeBar's update()
        self.WorkingData.Update(data.LastPrice, 0, 0, data.Quantity, 0, 0)

    def Scan(self, time):
        #Scans this consolidator to see if it should emit a bar due to the volume

        if self.barSize is not None and self.WorkingData is not None:
            if self.WorkingData.Volume >= self.barSize:

                #Trigger the event handler with a copy of self and the data
                self.OnDataConsolidated(self, self.WorkingData)

                #Set the most recent consolidated piece of data and then clear the workingData
                self.Consolidated = self.WorkingData
                self.WorkingData = None