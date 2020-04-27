import numpy as np
from common import Strategy


class EpsilonGreedyStrategy(Strategy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, state, Q, env):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, env.action_space.n)
        else:
            return np.argmax(Q[state[0], state[1]])