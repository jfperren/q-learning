import numpy as np
from agents.exploration.strategy import ExplorationStrategy


class EpsilonGreedyStrategy(ExplorationStrategy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def explore(self, state):
        return np.random.random() < self.epsilon