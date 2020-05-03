
import copy
from collections import deque
import numpy as np
import torch
import random

from agents.behavior.exploration_behavior import ExplorationBehavior


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
        max_grad_norm=1,
        skip=20,
        target_update_frequency=10
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
        self.max_grad_norm = max_grad_norm
        self.memory = deque(maxlen=memory_size)
        self.epoch = 0
        self.target_update_frequency = target_update_frequency
        
        if target_update_frequency is not None:
            self.target_model = copy.deepcopy(self.model)
        else:
            self.target_model = None

    def should_skip(self, epoch):
        return self.skip is not None and epoch % self.skip != 0
    
    def has_target_network(self):
        return self.target_update_frequency is not None
    
    def should_synchronize_networks(self, epoch):
        return self.has_target_network() and epoch % self.target_update_frequency == 0
    
    def synchronyze_networks(self):
        if self.target_model is not None:
            self.target_model = copy.deepcopy(self.model)

    def forward(self, state, use_target=False):
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        if use_target:
            return self.target_model.forward(torch.Tensor(state))
        else:
            return self.model.forward(torch.Tensor(state))

    def predict(self, state, use_target=False):
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        with torch.no_grad():
            if use_target:
                return self.target_model(torch.Tensor(state))
            else:
                return self.model(torch.Tensor(state))

    def best_action(self, state):
        # Best action always uses the current network, not the target one.
        return self.predict(state, use_target=False).argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def on_episode_end(self, context):
        super().on_episode_end(context)
        
        if self.should_synchronize_networks(context.get_episode_value('epoch')):
            self.synchronyze_networks()

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
            # If we have a target network, this is the one that should be used
            # for estimating the future rewards.
            target = self.predict(state, use_target=self.has_target_network())
            
            if not done:
                future_reward = self.predict(next_state, use_target=self.has_target_network()).max().item()
                target[0][action] = reward + self.discount * future_reward
            else:
                target[0][action] = reward

            # Move towards target with one backward pass
            q_values = self.forward(state, use_target=False)
            loss = self.loss(q_values, target)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm is not None:        
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)   
                context.append_episode_value('gradient_norm', norm)
                
#             if loss.detach() > 500:
#                 print('BEFORE STEP')
#                 print('epoch: {}'.format(context.get_episode_value('epoch')))
#                 print('loss: {}'.format(loss.detach()))
#                 print('state: {}'.format(state))
#                 print('action: {}'.format(action))
#                 print('next state: {}'.format(next_state))
#                 print('done: {}'.format(done))
#                 print('q_values: {}'.format(q_values))
#                 print('target: {}'.format(target))

            self.optimizer.step()
            
#             if loss.detach() > 500:
#                 target = self.predict(state)
#                 if not done:
#                     future_reward = self.predict(next_state, False).max().item()
#                     target[0][action] = reward + self.discount * future_reward
#                 else:
#                     target[0][action] = reward
#                 q_values = self.forward(state)
#                 loss = self.loss(q_values, target)
#                 print('AFTER STEP')
#                 print('epoch: {}'.format(context.get_episode_value('epoch')))
#                 print('loss: {}'.format(loss.detach()))
#                 print('state: {}'.format(state))
#                 print('action: {}'.format(action))
#                 print('next state: {}'.format(next_state))
#                 print('done: {}'.format(done))
#                 print('q_values: {}'.format(q_values))
#                 print('target: {}'.format(target))
            
            context.append_episode_value('loss', loss.detach())
            context.append_episode_value('target', target[0][action].detach())