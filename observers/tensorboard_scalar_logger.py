from common import (
    LearningObserver,
)


class TensorboardScalarLogger(LearningObserver):
    
    def __init__(self, tb, name, apply):
        self.tb = tb
        self.name = name
        self.apply = apply
        
    def on_episode_end(self, context):
        value = self.apply(context)
        if value is not None:
            self.tb.log_scalar(
                self.name, 
                value,
                context.get_episode_value('epoch')
            )