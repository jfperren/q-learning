class FrequencyTrigger():
    
    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency
        
    def should_trigger(self, epoch):
        return epoch % self.frequency == 0