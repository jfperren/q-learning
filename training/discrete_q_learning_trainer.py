import numpy as np
import time


class DiscreteQLearningTrainer:
            
    def __init__(self, env, discretizer, training_config, epsilon, observers=[]):
        
        self.env = env
        self.discretizer = discretizer
        self.training_config = training_config
        self.epsilon = epsilon
        self.observers = observers
        
        self.resetQ()
    
    def discretize(self, s):
        return self.discretizer.parse(s)
    
    def resetQ(self):
        self.Q = np.zeros(
            np.concatenate([self.discretizer.dimensions, [self.env.action_space.n]])
        )
        
    def resetEnv(self):
        state = self.env.reset()
        return self.discretize(state)

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            return np.argmax(self.Q[state[0], state[1]])
        
    def step(self, state):
        
        # Choose an action
        action = self.epsilon_greedy(state)
        
        # Perform update step, don't forget to discretize
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.discretize(next_state)
        
        # Update Q-table
        future_reward = np.max(self.Q[next_state[0], next_state[1], :])
        delta = reward + self.training_config.discount * future_reward - self.Q[state[0], state[1], action]
        self.Q[state[0], state[1], action] += self.training_config.learning_rate * delta
        
        self.notifyObservers('on_step_end', {
            'action': action,
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'done': done,
        })
                                   
        return next_state, reward, done
    
    def render(self, fps=60):
        
        state = self.resetEnv()
        done = False
        total_reward = 0
        
        while not done:
            self.env.render()
            state, reward, done = self.step(state)
            total_reward += reward
            time.sleep(1/fps)
    
    def episode(self):
        
        state = self.resetEnv()
        done = False
        total_reward = 0
        
        while not done:
            state, reward, done = self.step(state)
            total_reward += reward
            
        self.epsilon -= self.epsilon / self.training_config.episodes

        return total_reward
            
    def train(self):
                
        for i in range(self.training_config.episodes):
            total_reward = self.episode()
            
            self.notifyObservers('on_episode_end', {
                'epoch': i,
                'reward': total_reward,
                'episodes': self.training_config.episodes,
                'epsilon': self.epsilon
            })

    def close(self):
        self.env.close()
                
    def notifyObservers(self, event, context):
        for observer in self.observers:
            getattr(observer, event)(context)
        