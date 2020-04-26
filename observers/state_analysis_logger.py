import numpy as np
from common import (
    FrequencyTrigger,
    LearningObserver,
)


class StateAnalysisLogger(LearningObserver, FrequencyTrigger):
    
    def __init__(self, env, discretizer, frequency):

        super().__init__(frequency)
        self.visit_stats = np.zeros(
            np.concatenate([discretizer.dimensions, [env.action_space.n]])
        )
        
    def on_step_end(self, context):
        self.visit_stats[context['state'][0], context['state'][1], context['action']] += 1
        
    def on_episode_end(self, context):
        if not self.should_trigger(context['epoch']):
            return 
        print('Visitation Pct: {}'.format(self.visited_pct()))
    
    def visited_pct(self):
        return np.sum(np.where(self.visit_stats != 0, 1, 0)) / self.visit_stats.size