

class QLearningContext:

    def __init__(self):
        self.session_values = {}
        self.episode_values = {}

    # -- Reset --

    def reset_episode_values(self):
        self.episode_values = {}

    # -- Setters --

    def set_session_value(self, key, value):
        self.session_values[key] = value

    def set_episode_value(self, key, value):
        self.episode_values[key] = value

    # -- Appenders --

    def append_session_value(self, key, value):
        if key not in self.session_values:
            self.session_values[key] = []
        self.session_values[key].append(value)

    def append_episode_value(self, key, value):
        if key not in self.episode_values:
            self.episode_values[key] = []
        self.episode_values[key].append(value)

    # -- Getters -- 

    def get_session_value(self, key):
        if key in self.session_values:
            return self.session_values[key]
        return None

    def get_episode_value(self, key):
        if key in self.episode_values:
            return self.episode_values[key]
        return None

    # -- Last Item Getters --

    def get_last_session_value(self, key):
        return self.session_values[key][-1]

    def get_last_episode_value(self, key):
        return self.episode_values[key][-1]

# -- Accessors --

def episode_value_accessor(key, aggregate=lambda x: x):
    def inner(c):
        if c.get_episode_value(key) is None:
            return None
        else:
            return aggregate(c.get_episode_value(key))

    return inner
