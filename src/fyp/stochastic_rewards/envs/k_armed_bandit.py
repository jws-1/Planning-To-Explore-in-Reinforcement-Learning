import numpy as np

class KArmedBandit:
    def __init__(self, k, seed=42, q_star=None):
        self.k = k
        np.random.seed(seed)
        if q_star is None:
            self.q_star = np.random.normal(0, 1, k) # true values of action preferences
        else:
            self.q_star = q_star
    
    def reset(self):
        self.q_star = np.random.normal(0, 1, self.k)
        self.action_count = np.zeros(self.k)
        self.total_reward = 0
        self.avg_reward = 0
    
    def get_reward(self, action):
        return np.random.normal(self.q_star[action], 1)
    
    # def choose_action(self, method, **kwargs):
    #     if method == "epsilon-greedy":
    #         epsilon = kwargs.get("epsilon", 0.1)
    #         if np.random.uniform() < epsilon:
    #             return np.random.choice(self.k)
    #         else:
    #             return np.argmax(self.avg_reward)
    #     elif method == "ucb":
    #         c = kwargs.get("c", 2)
    #         n = np.sum(self.action_count) + 1
    #         ucb = self.avg_reward + c*np.sqrt(np.log(n)/self.action_count)
    #         return np.argmax(ucb)
    
    # def update_reward(self, action, reward):
    #     self.action_count[action] += 1
    #     self.total_reward += reward
    #     self.avg_reward = self.total_reward / np.sum(self.action_count)
