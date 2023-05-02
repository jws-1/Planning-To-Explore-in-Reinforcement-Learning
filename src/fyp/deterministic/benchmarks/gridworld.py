import gym
from .benchmark_env import BenchmarkEnv
import numpy as np
from ..models import D_MDP
import os
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=(4,4), start_state=0, goal_state=15, walls=[]):
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal_state = goal_state
        self.walls = walls
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(np.prod(grid_size))
        self.nS = self.observation_space.n
        self.nA = self.action_space.n
        self.reset()


    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        row, col = divmod(self.current_state, self.grid_size[1])
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.grid_size[1] - 1, col + 1)
        elif action == 2: 
            row = min(self.grid_size[0] - 1, row + 1)
        elif action == 3: 
            col = max(0, col - 1)

        state = row * self.grid_size[1] + col
        if state not in self.walls:
            self.current_state = state

        return self.current_state, -1, self.current_state == self.goal_state, {}

class BenchmarkGridworld(BenchmarkEnv):

    def __init__(self):
        self.env_cls = GridWorldEnv
        self.env_kwargs = {"grid_size":(10,10), "start_state":95, "goal_state":55, "walls": [63, 63, 64, 66, 67, 68, 73, 83, 78, 88]}

    def reset_env(self):
        self.env = self.env_cls(**self.env_kwargs)
        self.env.max_reward = -4.0


    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):

        T = np.zeros((self.env.nS, self.env.nA), dtype=int)
        R = np.full((self.env.nS, self.env.nA), -1., dtype=float)

        reasonable_meta_states = {}
        walls = [63, 64, 65, 66, 67, 68, 73, 83, 78, 88]

        for s in range(self.env.nS):

            reasonable_meta_states[s] = []  

            s2d = np.unravel_index(s, self.env.grid_size)
            up2d = s2d + np.array([-1, 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array(self.env.grid_size) - np.array([1,1]))
            up = np.ravel_multi_index(up2d, self.env.grid_size)
            reasonable_meta_states[s].append(up)

            right2d = s2d + np.array([0, 1])
            right2d = np.clip(right2d, np.array([0,0]), np.array(self.env.grid_size) - np.array([1,1]))
            right = np.ravel_multi_index(right2d, self.env.grid_size)
            reasonable_meta_states[s].append(right)


            down2d = s2d + np.array([1, 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array(self.env.grid_size) - np.array([1,1]))
            down = np.ravel_multi_index(down2d,  self.env.grid_size)
            reasonable_meta_states[s].append(down)

            left2d = s2d + np.array([0, -1])
            left2d = np.clip(left2d, np.array([0,0]), np.array(self.env.grid_size) - np.array([1,1]))
            left = np.ravel_multi_index(left2d, self.env.grid_size)
            reasonable_meta_states[s].append(left)

            if up in walls:
                up = s
            if right in walls:
                right = s
            if down in walls:
                down = s
            if left in walls:
                left = s

            T[s, 0] = up
            T[s, 1] = right
            T[s, 2] = down
            T[s, 3] = left

        model = D_MDP(np.array(range(self.env.nS)), np.array([self.env.goal_state]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner, undiscretize_fn=lambda x : np.unravel_index(x, self.env.grid_size))
        return model


    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "deterministic")):
            os.mkdir(os.path.join("results", "deterministic"))
        if not os.path.exists(os.path.join("results", "deterministic", "gridworld")):
            os.mkdir(os.path.join("results", "deterministic", "gridworld"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states, regrets, regrets_95pc = result
            np.save((os.path.join("results", "deterministic", "gridworld", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "deterministic", "gridworld", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "deterministic", "gridworld", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "deterministic", "gridworld", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "deterministic", "gridworld", f"{agent}-regrets_95pc.npy")), regrets_95pc)