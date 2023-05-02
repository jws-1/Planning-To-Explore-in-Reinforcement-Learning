from .benchmark_env import BenchmarkEnv
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import gym_windy_gridworlds
from ..models import MDP


class BenchmarkWindyGridworld(BenchmarkEnv):

    def __init__(self, seed=42):
        self.env_name = "StochWindyGridWorld-v0"
        self.env_kwargs = {"SIMULATOR_SEED": 433}
        self.seed = seed

    def reset_env(self):
        self.env = gym.make(self.env_name, **self.env_kwargs)
        self.env.max_reward = -15.0
        self.env.start = 37

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        """
        Generates an inaccurate model for the MB agents.
        The inaccuracy is that the wind covers more columns than expected.
        """
        T = np.zeros((self.env.nS, self.env.nA, self.env.nS), dtype=int)
        R = np.full((self.env.nS, self.env.nA, self.env.nS), -1., dtype=float)

        reasonable_meta_states = {}

        wind = {
            0 : -3,
            1 : -2,
            2 : -4,
            3 : -3,
            4 : -1,
            5 : 1,
            6 : -2,
            7 : -2,
            8 : -1,
            9 : 0
        }

        for s in range(self.env.nS):

            reasonable_meta_states[s] = []

            s2d = np.unravel_index(s, (7,10))
            up2d = s2d + np.array([-1, 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array([6,9]))
            up = np.ravel_multi_index(up2d, (7,10))
            reasonable_meta_states[s].append(up)
            up2d+= np.array([wind[s2d[1]], 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array([6,9]))
            up = np.ravel_multi_index(up2d, (7,10))

            right2d = s2d + np.array([0, 1])
            right2d = np.clip(right2d, np.array([0,0]), np.array([6,9]))
            right = np.ravel_multi_index(right2d, (7,10))
            reasonable_meta_states[s].append(right)
            right2d+= np.array([wind[s2d[1]], 0])
            right2d = np.clip(right2d, np.array([0,0]), np.array([6,9]))
            right = np.ravel_multi_index(right2d, (7,10))

            down2d = s2d + np.array([1, 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array([6,9]))
            down = np.ravel_multi_index(down2d,  (7,10))
            reasonable_meta_states[s].append(down)
            down2d+= np.array([wind[s2d[1]], 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array([6,9]))
            down = np.ravel_multi_index(down2d,  (7,10))       

            left2d = s2d + np.array([0, -1])
            left2d = np.clip(left2d, np.array([0,0]), np.array([6,9]))
            left = np.ravel_multi_index(left2d, (7,10))
            reasonable_meta_states[s].append(left)
            left2d+= np.array([wind[s2d[1]], 0])
            left2d = np.clip(left2d, np.array([0,0]), np.array([6,9]))
            left = np.ravel_multi_index(left2d, (7,10))

            T[s, 0, up] = 1.0
            T[s, 1, right] = 1.0
            T[s, 2,down] = 1.0
            T[s, 3, left] = 1.0

        model = MDP(np.array(range(self.env.nS)), np.array([37]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner, undiscretize_fn=lambda x: np.unravel_index(x, (7,10)))
        return model

    def generate_reasonable_meta(self):
        pass

    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "stochastic")):
            os.mkdir(os.path.join("results", "stochastic"))
        if not os.path.exists(os.path.join("results", "stochastic", "windy_gridworld")):
            os.mkdir(os.path.join("results", "stochastic", "windy_gridworld"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states, states_95pc, regrets, regrets_95pc = result
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-states_95pc.npy")), states_95pc)
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "stochastic", "windy_gridworld", f"{agent}-regrets_95pc.npy")), regrets_95pc)
