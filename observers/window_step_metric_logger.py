import numpy as np
from common import (
    FrequencyTrigger,
    LearningObserver,
)


class WindowStepMetricLogger(LearningObserver, FrequencyTrigger):
    
    def __init__(self, window_size, metric):
        
        super().__init__(window_size)
        self.metric = metric
        self.accumulated = []
        self.window_metrics = []
    
    def on_step_end(self, context):
        if self.metric in context:
            self.accumulated.append(context[self.metric])
        
    def on_episode_end(self, context):
        if self.should_trigger(context['epoch']):
            if len(self.accumulated) != 0:
                mean_metric = np.mean(self.accumulated)
                self.window_metrics.append(mean_metric)
                self.accumulated = []
            else:
                mean_metric = None
            print("Epoch: {} | Window {}: {}".format(context['epoch'], self.metric, mean_metric))