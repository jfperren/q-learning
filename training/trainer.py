import numpy as np
import time
import datetime as dt
from training.training_context import QLearningContext


class QLearningTrainer:
            
    def __init__(self, env, agent):
        
        self.env = env
        self.agent = agent
        self.context = QLearningContext()
            
    def step(self, state):
        
        # Choose an action
        action = self.agent.act(state)
        
        # Perform update step
        next_state, reward, done, _ = self.env.step(action)
        
        # Update Q-table
        self.agent.learn(state, action, reward, next_state, done, self.context)

        # Update Context & Notify Observers
        self.context.append_episode_value('action', action)
        self.context.append_episode_value('reward', reward)
        self.context.append_episode_value('next_state', next_state)
        self.context.append_episode_value('done', done)
        self.notifyObservers('on_step_end')
                                   
        return next_state, reward, done
    
    def render(self, fps=60):
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            self.env.render()
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            time.sleep(1/fps)
    
    def episode(self, epoch):

        # Reset Context & Call Observers
        self.context.reset_episode_values()
        self.context.set_episode_value('epoch', epoch)
        self.notifyObservers('on_episode_start')
        
        state = self.env.reset()
        done = False

        while not done:
            state, reward, done = self.step(state)

        self.notifyObservers('on_episode_end')
            
    def train(self, epochs, observers):

        self.observers = [self.agent] + observers

        self.context.set_session_value('start_time', dt.datetime.now())
        self.context.set_session_value('epochs', epochs)
        self.notifyObservers('on_train_start')
                
        for i in range(epochs):
            self.episode(i)
            
    def close(self):
        self.env.close()
                
    def notifyObservers(self, event):
        for observer in self.observers:
            getattr(observer, event)(self.context)
        