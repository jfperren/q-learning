import numpy as np
from common import (
    FrequencyTrigger,
    LearningObserver,
)


class WindowMetricLogger(LearningObserver, FrequencyTrigger):
    
    def __init__(self, window_size, metric):
        
        super().__init__(window_size)
        self.metric = metric
        self.accumulated = []
        self.window_metrics = []
    
    def on_episode_end(self, context):
        self.accumulated.append(context[self.metric])
        
        if self.should_trigger(context['epoch']):
            mean_metric = np.mean(self.accumulated)
            self.window_metrics.append(mean_metric)
            self.accumulated = []
            print("Epoch: {} | Window {}: {}".format(context['epoch'], self.metric, mean_metric))