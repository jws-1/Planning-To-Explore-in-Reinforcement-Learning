
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
from copy import deepcopy
from types import SimpleNamespace

class RLAgent():

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.Q = {state : {action : 0. for action in range(self.env.nA)} for state in range(self.env.nS)}

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))

        if config.decay:
            decay_factor = (config.eps_min/config.eps)**(1/config.episodes)
        else:
            decay_factor = 1.0
        
        eps = config.eps

        for i in range(config.episodes):
            # if i % (config.episodes // 100) == 0:
            print(f"RL-AGENT: episode {i}")

            done = False
            state = self.env.reset()

            while not done:
                
                if random.uniform(0, 1) < eps:
                    action = self.env.action_space.sample()
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])

                next_state, reward, done, _ = self.env.step(action)

                old_value = self.Q[state][action]
                next_max = max(self.Q[next_state].values())
                new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                self.Q[state][action] = new_value

                if state != next_state:
                    states[i][state]+=1
                state = next_state
                rewards[i] += reward
                
            states[i][state]+=1
            eps = eps * decay_factor
        return rewards, states

    def learn_and_aggregate(self, config):
        rewards_windows = np.zeros((config.m, config.episodes - config.window_size+1, config.window_size))
        states = np.zeros((config.m, config.episodes, self.env.nS))
        for i in range(config.m):

            rewards, states_ = self.learn(config)

            rewards_window = np.lib.stride_tricks.sliding_window_view(rewards, config.window_size)
            rewards_windows[i] = deepcopy(rewards_window)
            
            states[i] = states_

        # Calculate 95% confidence interval for each window.

        aggregated_rewards_windows = np.mean(np.mean(rewards_windows, axis=2), axis=0)
        rewards_95pc = []
        for i in range(rewards_windows.shape[1]):
            a = []
            for j in range(rewards_windows.shape[0]):
                a.extend(list(np.ravel(rewards_windows[j][i])))
            rewards_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        return aggregated_rewards_windows, np.array(rewards_95pc), states

    def plot_results(self, rewards, states, policy="e-greedy", rewards_95pc=None, save=True, config=None):
        if not os.path.exists(policy) and save:
            os.mkdir(policy)
        print(rewards.shape[0]-50)
        plt.figure(0)
        plt.plot(rewards)
        plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"RL Agent Reward/Episode {(policy)}")
        if save:
            plt.savefig(os.path.join(policy, f"{policy}-rewards.png"))
            plt.clf()