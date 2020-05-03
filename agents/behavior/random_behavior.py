import numpy as np


class RandomBehavior:

    def __init__(self, env):
        self.env = env

    def random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
