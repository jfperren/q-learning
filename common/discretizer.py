import numpy as np


class Discretizer():
    
    def __init__(self, env, precision):
        
        if len(precision) != env.observation_space.shape[0]:
            raise Error("Buckets should have same shape as observation space. {} vs {}".format(
                len(precision), env.observation_space.shape[0]
            ))
        
        self.bucket_starts = env.observation_space.low
        self.precision = precision
        self.dimensions = ((env.observation_space.high - env.observation_space.low) / precision).round().astype(int) + 1
        
    def parse(self, state):   
        return np.array([int(s) for s in (state - self.bucket_starts) / self.precision])