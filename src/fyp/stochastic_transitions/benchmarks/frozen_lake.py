# from ..agents import MetaPRLAgent, PRLAgent, RLAgent
# import gym
# import numpy as np
# from ..models import MDP
# from types import SimpleNamespace
# from plot import plot_results
# from collections import defaultdict
# def generate_inaccurate_mdp(env, mdp):
#     return mdp

from ..models import MDP
import gym
from .benchmark_env import BenchmarkEnv
import os
import numpy as np
import matplotlib.pyplot as plt
class BenchmarkFrozenLake(BenchmarkEnv):
    
    def __init__(self):
        self.env_name = "FrozenLake-v1"
        self.env_kwargs = {"is_slippery":True, "map_name" : "4x4"}

    def reset_env(self):
        self.env = gym.make(self.env_name, **self.env_kwargs)
        self.env.seed(442)
        self.env.max_reward = 1.0
        self.env.start = 0

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        """
        Make this inaccurate, in some way.
        """
        T = np.zeros((self.env.nS, self.env.nA, self.env.nS), dtype=int)
        R = np.full((self.env.nS, self.env.nA, self.env.nS), -1., dtype=float)

        reasonable_meta_states = {}

        for s in range(self.env.nS):

            reasonable_meta_states[s] = []

            s2d = np.unravel_index(s, (4,4))
            up2d = s2d + np.array([-1, 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array([3,3]))
            up = np.ravel_multi_index(up2d, (4,4))
            reasonable_meta_states[s].append(up)

            right2d = s2d + np.array([0, 1])
            right2d = np.clip(right2d, np.array([0,0]), np.array([3,3]))
            right = np.ravel_multi_index(right2d, (4,4))
            reasonable_meta_states[s].append(right)

            down2d = s2d + np.array([1, 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array([3,3]))
            down = np.ravel_multi_index(down2d,  (4,4))
            reasonable_meta_states[s].append(down)

            left2d = s2d + np.array([0, -1])
            left2d = np.clip(left2d, np.array([0,0]), np.array([3,3]))
            left = np.ravel_multi_index(left2d, (4,4))
            reasonable_meta_states[s].append(left)

            T[s, 3, up] = 1.0
            T[s, 2, right] = 1.0
            T[s, 1, down] = 1.0
            T[s, 0, left] = 1.0


        model = MDP(np.array(range(self.env.nS)), np.array([15]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner, undiscretize_fn=lambda x: np.unravel_index(x, (4,4)))
        return model


    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "stochastic")):
            os.mkdir(os.path.join("results", "stochastic"))
        if not os.path.exists(os.path.join("results", "stochastic", "frozen_lake")):
            os.mkdir(os.path.join("results", "stochastic", "frozen_lake"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states, states_95pc, regrets, regrets_95pc = result
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-states_95pc.npy")), states_95pc)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-regrets_95pc.npy")), regrets_95pc)