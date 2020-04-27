import numpy as np
from common import Strategy
from strategies.epsilon_greedy import EpsilonGreedyStrategy


class EpsilonDecreasingStrategy(EpsilonGreedyStrategy):

    def __init__(self, epsilon):
        super().__init__(epsilon)

    def on_episode_end(self, context):
        self.epsilon -= self.epsilon / context['episodes']