
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
        self.Q = np.zeros((self.env.nS, self.env.nA))

    def learn(self, config):
        self.reset()
        actions = {i: [] for i in range(config.episodes)}

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))

        if config.decay:
            decay_factor = (config.eps_min/config.eps)**(1/config.episodes)
        else:
            decay_factor = 1.0
        
        eps = config.eps

        for i in range(config.episodes):
            # if i % (config.episodes // 100) == 0:
            #     print(f"RL-AGENT: episode {i}")

            done = False
            state = self.env.reset()
            
            while not done:
                if random.uniform(0, 1) < eps:
                    action = self.env.action_space.sample()
                else:
                    max_Q = np.max(self.Q[state])
                    max_actions = [a for a in range(self.env.nA) if self.Q[state][a] == max_Q]
                    action = np.random.choice(max_actions)
                actions[i].append(action)
                next_state, reward, done, info = self.env.step(action)
                # if done and not info.get("TimeLimit.truncated"):
                #     print("Completed ", i)
                self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + np.max(self.Q[next_state])) - self.Q[state][action])

                if state != next_state:
                    states[i][state]+=1
                state = next_state
                rewards[i] += reward
                
            states[i][state]+=1
            eps = eps * decay_factor
        

        print("*"*20, "e","*"*20)
        print(self.Q)
        print(actions[config.episodes-1])
        print("*"*20, "e","*"*20)
        return rewards, states

    def learn_and_aggregate(self, config):
        rewards_windows = np.zeros((config.m, config.episodes - config.window_size+1, config.window_size))
        states_windows = np.zeros((config.m, config.episodes - config.window_size+1, self.env.nS, config.window_size))
        regrets_windows = np.zeros((config.m, config.episodes - config.window_size+1, config.window_size))
        
        for i in range(config.m):
            rewards, states_ = self.learn(config)

            # Compute cumulative rewards over episodes
            rewards_window = np.lib.stride_tricks.sliding_window_view(rewards, config.window_size)
            rewards_windows[i] = deepcopy(rewards_window)

            # Compute cumulative regrets over episodes
            max_rewards = np.full_like(rewards, self.env.max_reward)
            regrets = max_rewards - rewards
            regrets_window = np.lib.stride_tricks.sliding_window_view(regrets, config.window_size)
            regrets_windows[i] = deepcopy(regrets_window)

            print(states_.shape)
            # Compute states over episodes
            states_window = np.lib.stride_tricks.sliding_window_view(states_, config.window_size, axis=0)
            print(states_window.shape, states_windows.shape, states_windows[i].shape)
            states_windows[i] = deepcopy(states_window)

        # Compute confidence intervals for rewards over windows
        rewards_95pc = []
        for i in range(rewards_windows.shape[1]):
            a = []
            for j in range(rewards_windows.shape[0]):
                a.extend(list(np.ravel(rewards_windows[j][i])))
            rewards_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        # Compute confidence intervals for states over windows
        states_95pc = []
        for i in range(states_windows.shape[1]):
            a = []
            for j in range(states_windows.shape[0]):
                a.extend(list(np.ravel(states_windows[j][i])))
            states_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        # Compute confidence intervals for regrets over windows
        regrets_95pc = []
        for i in range(regrets_windows.shape[1]):
            a = []
            for j in range(regrets_windows.shape[0]):
                a.extend(list(np.ravel(regrets_windows[j][i])))
            regrets_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        # Compute mean of aggregated rewards over all runs and windows
        aggregated_rewards_windows = np.mean(np.mean(rewards_windows, axis=2), axis=0)

        # Compute mean of aggregated states over all runs and windows
        aggregated_states_windows = np.mean(states_windows, axis=(0,2))

        # Compute mean of aggregated regrets over all runs and windows
        aggregated_regrets_windows = np.mean(np.mean(regrets_windows, axis=2), axis=0)

        return aggregated_rewards_windows, np.array(rewards_95pc), aggregated_states_windows, np.array(states_95pc), aggregated_regrets_windows, np.array(regrets_95pc)



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