import gym
from .benchmark_env import BenchmarkEnv
from ..models import D_MDP
import numpy as np
import itertools
import math
import os
import matplotlib.pyplot as plt

class MountainCarDiscretized(gym.Env):
    def __init__(self, n_bins=(400,400)):
        self.env = gym.make("MountainCar-v0")
        self.nA = self.env.action_space.n
        self.action_space = self.env.action_space
        self.n_bins = n_bins
        self.grid = [
            np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], n_bins[0], endpoint=False)[1:],
            np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], n_bins[1], endpoint=False)[1:]
        ]
        print(self.grid)
        print(len(self.grid[0]), len(self.grid[1]))
        self.nS = (self.n_bins[0]-1) * (self.n_bins[1]-1)
        print(self.nS)

    def discretize(self, obs):
        x = int(np.digitize(obs[0], self.grid[0]))
        y = int(np.digitize(obs[1], self.grid[1]))
        x = min(x, self.n_bins[0]-2)
        y = min(y, self.n_bins[1]-2)
        # print(x, y)
        return np.ravel_multi_index((x,y), (len(self.grid[0]), len(self.grid[1])))
        # return int(np.digitize(obs[0], self.grid[0])) * self.n_bins[0] + int(np.digitize(obs[1], self.grid[1]))

    def undiscretize(self, idx):
        return np.unravel_index(idx, (len(self.grid[0]), len(self.grid[1])))
        # x_idx = idx // self.n_bins[1]
        # y_idx = idx % self.n_bins[1]
        # x = self.grid[0][x_idx]
        # y = self.grid[1][y_idx]
        # return np.array([x, y])

    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        return self.discretize(obs), reward, done, info

    # def discretize(self, obs):
    #     env_low = self.env.observation_space.low
    #     env_high = self.env.observation_space.high
    #     env_dx = (env_high - env_low) / self.n_bins
    #     a = int((obs[0] - env_low[0])/env_dx[0])
    #     b = int((obs[1] - env_low[1])/env_dx[1])
    #     return a*self.n_bins + b

    # def undiscretize(self, obs):
    #     a, b = obs // self.n_bins, obs % self.n_bins
    #     env_low = self.env.observation_space.low
    #     env_high = self.env.observation_space.high
    #     env_dx = (env_high - env_low) / self.n_bins
    #     x = env_low[0] + (b + 0.5) * env_dx[0]
    #     y = env_low[1] + (a + 0.5) * env_dx[1]
    #     return x, y

    def reset(self):
        obs = self.env.reset()
        return self.discretize(obs)

    def render(self, mode="human"):
        return self.env.render(mode=mode)
    
    # def seed(self, seed):
    #     pass
    #     self.env.seed(seed)
    
class BenchmarkMountainCar(BenchmarkEnv):

    def __init__(self, seed=42):
        self.env_cls = MountainCarDiscretized
        self.seed = seed
    
    def reset_env(self):
        self.env = self.env_cls()
        self.env.seed(self.seed)

    def generate_model(self, reasonable_meta=False, noise=False):
        """
        Generates an inaccurate model for the MB agents.
        The inaccuracy is that the force and gravity are misrepresented.
        """
        # max_position, max_velocity = self.env.env.observation_space.high
        # goal_positions = np.arange(0.5, max_position, 0.1)
        # goal_velocities = np.arange(0, max_velocity, 0.01)
        # print(goal_positions, goal_velocities)
        # goal_states = np.array([self.env.discretize((pos, vel)) for pos, vel in itertools.product(goal_positions, goal_velocities)])
        goal_states = np.array([self.env.discretize((0.50100002, 0.0))])
        T = np.zeros((self.env.nS, self.env.nA), dtype=int)
        R = np.full((self.env.nS, self.env.nA), -1.0, dtype=float)

        reasonable_meta_states = {}

        for s in range(self.env.nS):
            # print(s)
            reasonable_meta_states[s] = [max(s-1, 0), min(s+1, self.env.nS-1)]
            for a in range(self.env.nA):

                position, velocity = self.env.undiscretize(s)

                velocity += self.env.env.force * a - self.env.env.gravity * math.cos(3*position)#(a - 1) * self.env.env.force - math.cos(3 * position) * (self.env.env.gravity)
                velocity = np.clip(velocity, -self.env.env.max_speed, self.env.env.max_speed)

                position += velocity
                position = np.clip(position, self.env.env.min_position, self.env.env.max_position)

                next_s = self.env.discretize((position, velocity))
                T[s, a] = next_s

        print(goal_states)
        print(T[0])
        model = D_MDP(np.array(range(self.env.nS)), goal_states, np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states)
        return model

    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "deterministic")):
            os.mkdir(os.path.join("results", "deterministic"))
        if not os.path.exists(os.path.join("results", "deterministic", "mountain_car")):
            os.mkdir(os.path.join("results", "deterministic", "mountain_car"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states = result
            np.save((os.path.join("results", "deterministic", "mountain_car", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "deterministic", "mountain_car", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "deterministic", "mountain_car", f"{agent}-states.npy")), states)
            plt.figure(0)
            plt.plot(rewards, label=agent)
            plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Mountain Car {agent} Learning Curve")    
            plt.savefig((os.path.join("results", "deterministic", "mountain_car", f"{agent}-rewards.png")))
            plt.clf()

            plt.figure(1)
            plt.plot(rewards, label=agent)
            if not "PRL" in agent:
                print(f"[Mountain Car] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
            else:
                print(f"[Mountain Car] {agent} mean, std, min, max, final rewards: {np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards), rewards[-1]}")
                print(f"[Mountain Car] {agent} mean, std, min, max, final planning rewards: {np.mean(rewards[:p-w]), np.std(rewards[:p-w]), np.min(rewards[:p-w]), np.max(rewards[:p-w]), rewards[:p-w][-1]}")
            plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

        plt.figure(1)
        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Mountain Car")  
        plt.savefig((os.path.join("results", "deterministic", "mountain_car", "rewards.png")))
        plt.clf()