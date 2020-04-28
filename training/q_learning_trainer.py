import numpy as np
import time


class QLearningTrainer:
            
    def __init__(self, env, solver, strategy, observers=[]):
        
        self.env = env
        self.solver = solver
        self.strategy = strategy
        self.observers = observers + [strategy]
            
    def step(self, state):
        
        # Choose an action
        action = self.strategy.select_action(state, self.solver)
        
        # Perform update step, don't forget to discretize
        next_state, reward, done, _ = self.env.step(action)
        
        # Update Q-table
        self.solver.remember(state, action, reward, next_state)
        
        self.notifyObservers('on_step_end', {
            'action': action,
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'done': done,
        })
                                   
        return next_state, reward, done
    
    def render(self, fps=60):
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            self.env.render()
            state, reward, done = self.step(state)
            total_reward += reward
            time.sleep(1/fps)
    
    def episode(self):
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state, reward, done = self.step(state)
            total_reward += reward

        return total_reward
            
    def train(self, episodes):
                
        for i in range(episodes):
            total_reward = self.episode()
            
            self.notifyObservers('on_episode_end', {
                'epoch': i,
                'reward': total_reward,
                'episodes': episodes
            })

    def close(self):
        self.env.close()
                
    def notifyObservers(self, event, context):
        for observer in self.observers:
            getattr(observer, event)(context)
        