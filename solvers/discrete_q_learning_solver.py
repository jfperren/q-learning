import numpy as np


class DiscreteQLearningSolver:

    def __init__(self, env, discretizer, learning_rate, discount):
        self.env = env
        self.discretizer = discretizer
        self.learning_rate = learning_rate
        self.discount = discount
        self.setup()

    def setup(self):
        self.Q = np.zeros(
            np.concatenate([self.discretizer.dimensions, [self.env.action_space.n]])
        )
    
    def discretize(self, state):
        return tuple(self.discretizer.parse(state))

    def best_action(self, state):
        state = self.discretize(state)
        return np.argmax(self.Q[state])

    def random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def remember(self, state, action, reward, next_state):
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        future_reward = np.max(self.Q[next_state])
        delta = reward + self.discount * future_reward - self.Q[state + (action,)]
        self.Q[state + (action,)] += self.learning_rate * delta
    