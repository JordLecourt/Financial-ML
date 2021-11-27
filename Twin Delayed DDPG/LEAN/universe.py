from QuantConnect.Data.UniverseSelection import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

class LiquidValueUniverseSelectionModel(FundamentalUniverseSelectionModel):
    
    def __init__(self):
        super().__init__(True, None)
        self.lastMonth = -1

        
    def SelectCoarse(self, algorithm, coarse):
        # Change the universe only once a month
        if self.lastMonth == algorithm.Time.month:
            return Universe.Unchanged
        
        # Sort Symbol by Dollar Volume
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        sortedByPrice = [x for x in sortedByDollarVolume if (x.Price > 10)]# and x.Price < 20
        
        # Return the fist 10
        return [x.Symbol for x in sortedByPrice[:100]]
        
    def SelectFine(self, algorithm, fine):
        # Change the universe only once a month
        if self.lastMonth == algorithm.Time.month:
            return Universe.Unchanged
        self.lastMonth = algorithm.Time.month
        
        # I don't use the true ATRP formula, only (high - low) / close
        atrp = {}
        
        for security in fine:    
            symbol = security.Symbol
            
            if symbol not in atrp:
                atrp[symbol] = SimpleMovingAverage(14)
                history = algorithm.History(symbol, 14, Resolution.Daily)
                for bar in history.itertuples():
                    atrp[symbol].Update(bar.Index[1], (bar.high - bar.low) / bar.close)
                
        # Sort symbol by ATR
        sortedByATR = sorted(fine, key=lambda x: atrp[x.Symbol].Current.Value, reverse=True)
            
        return [x.Symbol for x in sortedByATR[:1]] 