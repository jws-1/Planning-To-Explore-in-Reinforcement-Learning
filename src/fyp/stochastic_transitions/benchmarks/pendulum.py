import gym
import numpy as np
import itertools
from ..models import MDP

class DiscretizedPendulum(gym.Env):
    def __init__(self, num_theta_bins=20, num_theta_dot_bins=50, num_action_bins=10, theta_min=-np.pi, theta_max=np.pi, theta_dot_min=-8, theta_dot_max=8, action_min=-2, action_max=2):
        self.env = gym.make('Pendulum-v1')
        self.action_space = gym.spaces.Box(low=action_min, high=action_max, shape=(1,), dtype=np.float32)
        self.nA = num_action_bins

        self.num_theta_bins = num_theta_bins
        self.num_theta_dot_bins = num_theta_dot_bins
        self.num_action_bins = num_action_bins

        self.theta_bin_size = (theta_max - theta_min) / num_theta_bins
        self.theta_dot_bin_size = (theta_dot_max - theta_dot_min) / num_theta_dot_bins
        self.action_bin_size = (action_max - action_min) / num_action_bins

        self.nSTheta = num_theta_bins
        self.nSThetaDot = num_theta_dot_bins
        self.nSA = num_action_bins
        self.nS = self.nSTheta * self.nSThetaDot * self.nSA

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_dot_min = theta_dot_min
        self.theta_dot_max = theta_dot_max
        self.action_min = action_min
        self.action_max = action_max

    def discretize(self, obs):
        theta = obs[0]
        theta_dot = obs[1]
        theta_bin = int((theta - self.theta_min) // self.theta_bin_size)
        theta_dot_bin = int((theta_dot - self.theta_dot_min) // self.theta_dot_bin_size)
        theta_bin = np.clip(theta_bin, 0, self.nSTheta - 1)
        theta_dot_bin = np.clip(theta_dot_bin, 0, self.nSThetaDot - 1)
        return theta_bin * self.nSThetaDot * self.nSA + theta_dot_bin * self.nSA

    def from_flat_index(self, flat_index):
        theta_bin = flat_index // (self.nSThetaDot * self.nSA)
        theta = ((theta_bin + 0.5) * self.theta_bin_size) + self.theta_min
        theta_dot_bin = (flat_index % (self.nSThetaDot * self.nSA)) // self.nSA
        theta_dot = ((theta_dot_bin + 0.5) * self.theta_dot_bin_size) + self.theta_dot_min
        return theta, theta_dot

    def action_to_continuous(self, action_bin):
        return ((action_bin + 0.5) * self.action_bin_size) + self.action_min

    def step(self, action):
        action_bin = int((action - self.action_min) // self.action_bin_size)
        action_bin = np.clip(action_bin, 0, self.nSA - 1)
        obs, reward, done, info = self.env.step(self.action_to_continuous(action_bin))
        return self.discretize(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.discretize(obs)

    def render(self):
        return self.env.render()

def create_mdp(env):
    """
    Returns an instance of MDP, which corresponds to the env.
    """ 
    goal_states = np.array([env.discretize((0, 0))])

    transition_function = np.zeros((env.nS, env.nA, env.nS))
    reward_function = np.full((env.nS, env.nA, env.nS), 0.0)

    for s in range(env.nS):
        for a in range(env.nA):
            pos, vel = env.from_flat_index(s)

            action = (a - 1) * env.action_bin_size
            print(action)
            next_vel = vel + (-10.0/ env.env.length * np.sin(pos + np.pi) + action) * env.env.dt
            next_vel = np.clip(next_vel, -env.env.max_speed, env.env.max_speed)

            next_pos = pos + next_vel * env.env.dt
            next_pos = np.fmod(next_pos + np.pi, 2 * np.pi) - np.pi

            next_s = env.discretize((next_pos, next_vel))
            transition_function[s, a, next_s] = 1.0

            reward_function[s, a, next_s] = 1

    return MDP(np.array(range(env.nS)), goal_states, np.array(range(env.nA)), transition_function, reward_function)

create_mdp(DiscretizedPendulum())