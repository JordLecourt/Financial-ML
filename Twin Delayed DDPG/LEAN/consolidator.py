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
        # Check if the incoming data is a tick
        if data.DataType == MarketDataType.Tick:
            # If the tick bar is marked as suspicious, return
            if data.Suspicious:
                return
            
            tradebar = TradeBar(data.Time, data.Symbol, data.LastPrice, data.LastPrice,
                                data.LastPrice, data.LastPrice, data.Quantity)
        else:
            tradebar = TradeBar(data.Time, data.Symbol, data.Open, data.High,
                                data.Low, data.Close, data.Volume)
        
        #If we don't have bar yet, create one
        if self.WorkingData is None:
            self.WorkingData = tradebar
        else:
            #Update bar using TradeBar's update()
            lastTrade = (tradebar.Open + tradebar.High + tradebar.Low + tradebar.Close) / 4
            self.WorkingData.Update(lastTrade, 0, 0, tradebar.Volume, 0, 0)

    def Scan(self, time):
        #Scans this consolidator to see if it should emit a bar due to the volume

        if self.barSize is not None and self.WorkingData is not None:
            if self.WorkingData.Volume >= self.barSize:

                #Trigger the event handler with a copy of self and the data
                self.OnDataConsolidated(self, self.WorkingData)

                #Set the most recent consolidated piece of data and then clear the workingData
                self.Consolidated = self.WorkingData
                self.WorkingData = None