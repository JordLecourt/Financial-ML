from datetime import timedelta
from LEAN.universe import LiquidValueUniverseSelectionModel
from LEAN.alpha import RLAlphaModel

class RLAlgorithm(QCAlgorithm):
    """
    Initialization of the LEAN framework
    """
    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetEndDate(2021, 2, 1)
        self.SetCash(10000)
        
        # Universe Selection
        self.SetUniverseSelection(LiquidValueUniverseSelectionModel())
        self.UniverseSettings.Resolution = Resolution.Tick
        
        # Alpha Model
        self.SetAlpha(RLAlphaModel())
        
        # Portfolio construction
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Execution model
        self.SetExecution(ImmediateExecutionModel())

    def OnData(self, data):
        pass
        
       

        