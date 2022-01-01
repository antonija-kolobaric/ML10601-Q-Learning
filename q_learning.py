#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:31:16 2021

@author: antonijakolobaric
"""

import numpy as np
from environment import MountainCar
import sys

args = sys.argv

mode = args[1]
output_weights = args[2]
output_returns = args[3]
episodes = int(args[4])
max_iterations = int(args[5])
epsilon = float(args[6])
gamma = float(args[7])
lr = float(args[8])

env = MountainCar(mode=mode)


#Starter code taken from recitation 
class LinearModel:
    def __init__(self, state_size, action_size, lr, indices, w, b):
        """indices is True if indices are used as input for one-hot features.
        Otherwise, use the sparse representation of state as features
        self, state_size: int, action_size: int,
        lr: float, indices: bool)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.indices = indices
    
    def predict(self, state, w, b):
        """
        Given state, makes predictions.
        self, state: Dict[int, int]) -> List[float]:
        """

        current_q = w @ state + b
        return current_q
              
        
    def update(self, state, action, target, w, b):
        """
        Given state, action, and target, update weights.
        self, state: Dict[int, int], action: int, target: int
        """
        bias = self.lr * target
        b = b - bias

        w[action,:] = w[action,:] - (bias * state).T
        
        return w, b
        
        
class QLearningAgent:
    def __init__(self, env, mode, gamma, lr, epsilon):
        '''self, env: MountainCar, mode: str = None, gamma: float = 0.9,
        lr: float = 0.01, epsilon:float = 0.05)'''
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        
    def handle_mode(self, state):
        convert = np.zeros((self.env.state_space, 1))

        for key in state.keys():
            convert[key] = state[key]

        return convert
                
        
    def get_action(self, q):
        """epsilon-greedy strategy.
        Given state, returns action.
        self, state: Dict[int, int]) -> int
        """
        #95% of the time we choose the best action; 5% of the time choose random action --> aka exploring
        
        if np.random.rand() > self.epsilon:
            return np.argmax(q)
        else:
            return np.random.randint(3)

    def train(self, episodes, max_iterations):
        """training function.
        Train for ’episodes’ iterations, where at most ’max_iterations‘ iterations
        should be run for each episode. Returns a list of returns.
        self, episodes: int, max_iterations: int) -> List[float]
        """
        #Initialize stuff
        if mode == "raw":
            state_size = self.env.state_space
            indices = False
        if mode == "tile":
            state_size = self.env.state_space
            indices = True
        action_size = 3

            
        w = np.zeros((action_size,state_size))
        b = 0
        all_rewards = []
            
        lm = LinearModel(state_size, action_size, lr, indices, w, b)
        
        for episode in range(episodes):
            
            #accumulate episode reward
            episode_reward = 0
            
            #set up while loop
            done = False
            iteration = 0
            
            #reset state at the beginning
            state_dict = self.env.reset()
            current_state = self.handle_mode(state_dict)
            
            #iterate until you hit max iterations or you're in a terminate state
            while iteration < max_iterations and done == False:
                
                current_q = lm.predict(current_state, w, b)
                current_action = self.get_action(current_q)
                
                iteration +=1

                #use the environment given by TAs to get reward/done
                next_state_dict, current_reward, done = self.env.step(current_action)
                episode_reward += current_reward

                next_state = self.handle_mode(next_state_dict)
                next_q = lm.predict(next_state, w, b)
                next_action = self.get_action(next_q)

                target = current_q[current_action] - (current_reward + self.gamma * max(next_q))
                w, b = lm.update(current_state, current_action, target, w, b)
                
                current_state = next_state
                current_action = next_action

                if iteration == max_iterations or done == True:
                    all_rewards.append(episode_reward)

        return w, b, all_rewards
    
    def write_results(self, w, b, all_rewards):
        with open(output_weights, 'w') as f:
            for element in b:
                f.write(str(element))
                f.write("\n")
            for row in w.T:
                for column in row:
                    f.write(str(column))
                    f.write("\n")
                             
                
        with open(output_returns, 'w') as f:
            for element in all_rewards:
                f.write(str(element))
                f.write("\n")
    
agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
w, b, all_rewards = agent.train(episodes, max_iterations)
agent.write_results(w, b, all_rewards) 
