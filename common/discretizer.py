import numpy as np


class Discretizer(object):
    
    def __init__(self, env, precision, min_values=None, max_values=None):
        
        if len(precision) != env.observation_space.shape[0]:
            raise Error("Buckets should have same shape as observation space. {} vs {}".format(
                len(precision), env.observation_space.shape[0]
            ))
            
        if min_values != None and len(min_values) != env.observation_space.shape[0]:
            raise Error("Min Values should have same shape as observation space. {} vs {}".format(
                len(precision), env.observation_space.shape[0]
            ))
            
        if max_values != None and len(max_values) != env.observation_space.shape[0]:
            raise Error("Max Values should have same shape as observation space. {} vs {}".format(
                len(precision), env.observation_space.shape[0]
            ))
        
        self.min_values = min_values
        self.max_values = max_values
        
        low = self.clamp(env.observation_space.low)
        high = self.clamp(env.observation_space.high)
        
        self.bucket_starts = low
        self.precision = precision
        self.dimensions = ((high - low) / precision).round().astype(int) + 1
        
    def clamp(self, state):
        if self.min_values is not None:
            for i in range(state.shape[0]):
                if self.min_values[i] is not None:
                    state[i] = max(self.min_values[i], state[i])
                
        if self.max_values is not None:
            for i in range(state.shape[0]):
                if self.max_values[i] is not None:
                    state[i] = min(self.max_values[i], state[i])
        
        return state
        
    def parse(self, state):   
        return np.array([int(s) for s in (state - self.bucket_starts) / self.precision])