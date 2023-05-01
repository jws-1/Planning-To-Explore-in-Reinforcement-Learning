from .benchmark_env import BenchmarkEnv
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from ..models import D_MDP


class BenchmarkCliffWalking(BenchmarkEnv):

    def __init__(self, seed=42):
        self.env_name = "CliffWalking-v0"
        self.seed = seed

    def reset_env(self):
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.env.max_reward = -13.0

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        """
        Generates an inaccurate model for the MB agents.
        The inaccuracy is that the cliff is bigger in the model than in the real environment
        """
        T = np.zeros((self.env.nS, self.env.nA), dtype=int)
        R = np.full((self.env.nS, self.env.nA), -1., dtype=float)
        cliff_states = [
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        ]

        reasonable_meta_states = {}

        for s in range(self.env.nS):

            s2d = np.unravel_index(s, (4,12))

            up2d = s2d + np.array([-1, 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array([3,11]))
            up = np.ravel_multi_index(up2d, (4,12))

            right2d = s2d + np.array([0, 1])
            right2d = np.clip(right2d, np.array([0,0]), np.array([3,11]))
            right = np.ravel_multi_index(right2d, (4,12))

            down2d = s2d + np.array([1, 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array([3,11]))
            down = np.ravel_multi_index(down2d,  (4,12))

            left2d = s2d + np.array([0, -1])
            left2d = np.clip(left2d, np.array([0,0]), np.array([3,11]))
            left = np.ravel_multi_index(left2d, (4,12))

            reasonable_meta_states[s] = [up, right, down, left]


            if s not in cliff_states:
                up = 36 if up in cliff_states else up
                right = 36 if right in cliff_states else right
                down = 36 if down in cliff_states else down
                left = 36 if left in cliff_states else left

            T[s, 0] = up
            T[s, 1] = right
            T[s, 2] = down
            T[s, 3] = left
        
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                if T[s, a] in cliff_states:
                    R[s,a] = -100.
        model = D_MDP(np.array(range(self.env.nS)), np.array([47]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner, undiscretize_fn=lambda x : np.unravel_index(x, (4,12)))
        return model

    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "deterministic")):
            os.mkdir(os.path.join("results", "deterministic"))
        if not os.path.exists(os.path.join("results", "deterministic", "cliff_walking")):
            os.mkdir(os.path.join("results", "deterministic", "cliff_walking"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states, regrets, regrets_95pc = result
            np.save((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-regrets_95pc.npy")), regrets_95pc)

        #     plt.figure(0)
        #     plt.plot(rewards, label=agent)
        #     plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        #     plt.xlabel("Episode")
        #     plt.ylabel("Reward")
        #     plt.title(f"Cliff Walking {agent} Learning Curve")    
        #     plt.savefig((os.path.join("results", "deterministic", "cliff_walking", f"{agent}-rewards.png")))
        #     plt.clf()

        #     plt.figure(1)
        #     plt.plot(rewards, label=agent)
        #     if not "PRL" in agent:
        #         print(f"[Cliff Walking] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
        #     else:
        #         print(f"[Cliff Walking] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
        #         print(f"[Cliff Walking] {agent} mean, std, min, max, final planning rewards: {np.mean(rewards[:p-w]), np.std(rewards[:p-w]), np.min(rewards[:p-w]), np.max(rewards[:p-w]), rewards[:p-w][-1]}")
        #     plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

        # plt.figure(1)
        # plt.legend()
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.title("Cliff Walking")  
        # plt.savefig((os.path.join("results", "deterministic", "cliff_walking", "rewards.png")))
        # plt.clf()