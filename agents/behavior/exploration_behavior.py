from common.learning_observer import LearningObserver
from agents.behavior.random_behavior import RandomBehavior


class ExplorationBehavior(RandomBehavior, LearningObserver):

    def __init__(self, env, strategy):
        super().__init__(env)
        self.strategy = strategy

    def act(self, state):
        if self.strategy.explore(state):
            return self.random_action(state)
        else:
            return self.best_action(state)

    def best_action(self, state):
        raise NotImplementedError("Agent should define `best_action`.")

    def on_episode_end(self, context):
        self.strategy.on_episode_end(context)
