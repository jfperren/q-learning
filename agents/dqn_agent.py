
import copy
from collections import deque
import numpy as np
import torch
import random

from agents.behavior.exploration_behavior import ExplorationBehavior


class DQNAgent(ExplorationBehavior):

    def __init__(self, env, strategy, model, optimizer, loss, discount):
        super().__init__(env, strategy)
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.discount = discount

    def best_action(self, state):
        action = np.argmax(self.model.forward(torch.Tensor(state)).detach()).item()
        return action

    def learn(self, state, action, reward, next_state):

        # Initialize optimizer for this step
        self.optimizer.zero_grad()

        # Target is the current Q-Value with new reward for action taken
        out = self.model.forward(torch.Tensor(state))
        future_reward = self.model.forward(torch.Tensor(next_state)).detach().max().item()
        target = copy.deepcopy(out.detach())
        target[action] = reward + self.discount * future_reward

        # Move towards target with one backward pass
        loss = self.loss(out, target.detach())
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class DQNAgent(ExplorationBehavior):

    def __init__(
        self, 
        strategy,
        env, 
        model, 
        optimizer, 
        loss, 
        discount,
        memory_size=20000,
        batch_size=32,
        skip=20
    ):

        if skip == 0:
            raise Error("`skip` should either be None or > 1")

        super().__init__(env, strategy)
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.discount = discount
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.skip = skip
        self.memory = deque(maxlen=memory_size)
        self.epoch = 0

    def should_skip(self, epoch):
        return self.skip is not None and epoch % self.skip == 0

    def forward(self, state):
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        return self.model.forward(torch.Tensor(state))

    def predict(self, state):
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def best_action(self, state):
        return self.predict(state).argmax().item()
        
    def random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, state, action, reward, next_state, done, context):
        
        self.remember(state, action, reward, next_state, done)
        self.epoch = self.epoch + 1
        
        if len(self.memory) < self.batch_size:
            return
        
        if self.should_skip(context.get_episode_value('epoch')):
            return
        
        # Select a random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, state_next, done in batch:

            # Target is the current Q-Value with new reward for action taken
            target = self.predict(state)
            
            if not done:
                future_reward = self.predict(next_state).max().item()
                target[0][action] = reward + self.discount * future_reward
            else:
                target[0][action] = reward

            # Move towards target with one backward pass
            q_values = self.forward(state)
            loss = self.loss(q_values, target)
            
            self.optimizer.zero_grad()
            
            # for param in self.model.parameters():
            #     param.grad.data.clamp_(-1, 1)
            
            loss.backward()
            self.optimizer.step()
            
            context.append_episode_value('loss', loss.detach())
            context.append_episode_value('target', target[0][action].detach())