from .benchmark_env import BenchmarkEnv
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import gym_windy_gridworlds
from ..models import D_MDP


class BenchmarkWindyGridworld(BenchmarkEnv):

    def __init__(self, seed=42):
        self.env_name = "WindyGridWorld-v0"
        self.seed = seed

    def reset_env(self):
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.env.max_reward = -15.0

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        """
        Generates an inaccurate model for the MB agents.
        The inaccuracy is that the wind covers more columns than expected.
        """
        T = np.zeros((self.env.nS, self.env.nA), dtype=int)
        R = np.full((self.env.nS, self.env.nA), -1., dtype=float)

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

            T[s, 0] = up
            T[s, 1] = right
            T[s, 2] = down
            T[s, 3] = left


        model = D_MDP(np.array(range(self.env.nS)), np.array([37]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner,  undiscretize_fn=lambda x : np.unravel_index(x, (7,10)))
        return model

    def generate_reasonable_meta(self):
        pass

    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "deterministic")):
            os.mkdir(os.path.join("results", "deterministic"))
        if not os.path.exists(os.path.join("results", "deterministic", "windy_gridworld")):
            os.mkdir(os.path.join("results", "deterministic", "windy_gridworld"))

        for agent, result in results.items():
            rewards, rewards_95pc, states, regrets, regrets_95pc = result
            np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-regrets_95pc.npy")), regrets_95pc)

        # for agent, result in results.items():
        #     rewards, rewards_95pc, states = result
        #     np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards.npy")), rewards)
        #     np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards_95pc.npy")), rewards_95pc)
        #     np.save((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-states.npy")), states)
        #     plt.figure(0)
        #     plt.plot(rewards, label=agent)
        #     plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        #     plt.xlabel("Episode")
        #     plt.ylabel("Reward")
        #     plt.title(f"Cliff Walking {agent} Learning Curve")    
        #     plt.savefig((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards.png")))
        #     plt.clf()

        #     plt.figure(1)
        #     plt.plot(rewards, label=agent)
        #     if not "PRL" in agent:
        #         print(f"[Windy Gridworld] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
        #     else:
        #         print(f"[Windy Gridworld] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
        #         print(f"[Windy Gridworld] {agent} mean, std, min, max, final planning rewards: {np.mean(rewards[:p-w]), np.std(rewards[:p-w]), np.min(rewards[:p-w]), np.max(rewards[:p-w]), rewards[:p-w][-1]}")
        #     plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        
        # plt.figure(1)
        # plt.legend()
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.title("Windy Gridworld")  
        # plt.savefig((os.path.join("results", "deterministic", "windy_gridworld", "rewards.png")))
        # plt.clf()