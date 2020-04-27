from common.learning_observer import LearningObserver

class Strategy(LearningObserver):

    def select_action(self, state):
        raise NotImplementedError()