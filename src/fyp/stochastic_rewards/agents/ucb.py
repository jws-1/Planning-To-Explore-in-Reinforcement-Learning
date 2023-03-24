import numpy as np
from copy import deepcopy
import scipy.stats as st

class UCB():
    
    def __init__(self, bandit):
        self.bandit = bandit
        self.reset()
    
    def reset(self):
        self.N_a = np.zeros(self.bandit.k)
        self.total_reward = 0
        self.rewards = np.zeros(self.bandit.k)
        self.avg_reward = np.zeros(self.bandit.k)

    def choose_arm(self, config):
        n = np.sum(self.N_a)
        ucb_values = self.avg_reward + config.c * np.sqrt(np.log(n) / (self.N_a + 1e-5))
        return np.argmax(ucb_values)

    def learn(self, config):
        rewards = np.zeros(config.max_actions)
        arms = np.zeros((config.max_actions, self.bandit.k))

        for i in range(config.max_actions):
            arm = self.choose_arm(config)   
            reward = self.bandit.get_reward(arm)

            self.total_reward+=reward
            self.rewards[arm]+=reward
            self.N_a[arm]+=1
            self.avg_reward = self.rewards / self.N_a


            rewards[i] = reward
            arms[i][arm]+=1
        
        return rewards, arms

    def learn_and_aggregate(self, config):
        rewards_windows = np.zeros((config.m, config.max_actions - config.window_size+1, config.window_size))
        arms = np.zeros((config.m, config.max_actions, self.bandit.k))
        for i in range(config.m):

            rewards, actions_ = self.learn(config)
            
            rewards_window = np.lib.stride_tricks.sliding_window_view(rewards, config.window_size)
            rewards_windows[i] = deepcopy(rewards_window)

            arms[i] = actions_

        # Calculate 95% confidence interval for each window.

        aggregated_rewards_windows = np.mean(np.mean(rewards_windows, axis=2), axis=0)
        rewards_95pc = []
        for i in range(rewards_windows.shape[1]):
            a = []
            for j in range(rewards_windows.shape[0]):
                a.extend(list(np.ravel(rewards_windows[j][i])))
            rewards_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        return aggregated_rewards_windows, np.array(rewards_95pc), arms