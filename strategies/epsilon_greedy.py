import numpy as np
from common import Strategy


class EpsilonGreedyStrategy(Strategy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, state, solver):
        if np.random.random() < self.epsilon:
            return solver.random_action(state)
        else:
            return solver.best_action(state)