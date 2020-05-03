import numpy as np
from agents.exploration.epsilon_greedy import EpsilonGreedyStrategy


class EpsilonDecreasingStrategy(EpsilonGreedyStrategy):

    def __init__(self, initial_epsilon, min_epsilon, decay):
        super().__init__(initial_epsilon)
        self.min_epsilon = min_epsilon
        self.decay = decay

    def on_episode_end(self, context):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay)
        context.set_episode_value('epsilon', self.epsilon)