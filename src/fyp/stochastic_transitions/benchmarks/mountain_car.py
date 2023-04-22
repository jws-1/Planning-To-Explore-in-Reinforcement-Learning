from ..agents import MetaPRLAgent, PRLAgent, RLAgent
import gym
import numpy as np
from ..models import MDP
from types import SimpleNamespace
from plot import plot_results
from collections import defaultdict

mb_learn_config_dict = {
    "m": 1,
    "episodes": 1000,
    "window_size":1,
    "planning_steps":100,
    "eps": 0.0,
    "lr": 0.7,
    "min_lr":0.1,
    "decay_lr": True,
    "df": 1.0,
    "learn_model":True,
    "learn_meta_actions" : False
}

mb_config_dict = {
    "m": 5,
    "episodes": 50,
    "window_size":1,
    "planning_steps":100,
    "eps": 0.0,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":False,
}

mf_config_dict = {
    "m": 1,
    "episodes": 10000,
    "window_size":1,
    "eps": 0.9,
    "eps_min": 0.0,
    "decay": True,
    "lr": 0.2,
    "df": 1.0,
}

class MountainCarDiscretized(gym.Env):
    def __init__(self, n_bins=40):
        self.env = gym.make("MountainCar-v0")
        self.nA = self.env.action_space.n
        self.action_space = self.env.action_space
        self.n_bins = n_bins
        self.nS = n_bins**2

    def discretize(self, obs):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_dx = (env_high - env_low) / self.n_bins
        a = int((obs[0] - env_low[0])/env_dx[0])
        b = int((obs[1] - env_low[1])/env_dx[1])
        return a*self.n_bins + b

    def undiscretize(self, obs):
        a, b = obs // self.n_bins, obs % self.n_bins
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_dx = (env_high - env_low) / self.n_bins
        x = env_low[0] + (b + 0.5) * env_dx[0]
        y = env_low[1] + (a + 0.5) * env_dx[1]
        return x, y

    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        return self.discretize(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.discretize(obs)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

import math

def create_mdp(env): 
    """
    Returns an instance of MDP, which corresponds to the env.
    """
    import itertools
    max_position, max_velocity = env.env.observation_space.high
    goal_positions = np.arange(0.5, max_position+0.1, 0.1)
    goal_velocities = np.arange(0, max_velocity, 0.01)
    goal_states = np.array([env.discretize((pos, vel)) for pos, vel in itertools.product(goal_positions, goal_velocities)])

    transition_function = np.zeros((env.nS, env.nA, env.nS))
    reward_function = np.full((env.nS, env.nA, env.nS), -1.0)


    reasonable = defaultdict(list)

    # reward_function[:, :, goal_states] = 0.0
    for s in range(env.nS):
        reasonable[s] = [max(s-1, 0), min(s+1, env.nS-1)]
        for a in range(env.nA):
            
            position, velocity = env.undiscretize(s)

            velocity += (a - 1) * env.env.force - math.cos(3 * position) * (env.env.gravity)
            velocity = np.clip(velocity, -env.env.max_speed, env.env.max_speed)
            position += velocity
            position = np.clip(position, env.env.min_position, env.env.max_position)

            next_s = env.discretize((position, velocity))
    
            transition_function[s, a, next_s] = 1.0



    return MDP(np.array(range(env.nS)), goal_states, np.array(range(env.nA)), transition_function, reward_function, reasonable_meta_transitions=reasonable)

def benchmark(agent_cls, learn=True):
    np.random.seed(42)
    # Create an instance of the Mountain Car environment
    env = MountainCarDiscretized()

    if agent_cls in [MetaPRLAgent, PRLAgent]:
        inaccurate_mdp = create_mdp(env)
        agent = agent_cls(env, inaccurate_mdp)
        obs = env.reset()
        print(obs, env.undiscretize(obs))
        if learn:
            config = SimpleNamespace(**mb_learn_config_dict)
        else:
            config = SimpleNamespace(**mb_config_dict)
    else:
        agent = agent_cls(env)
        config = SimpleNamespace(**mf_config_dict)
    
    rewards, rewards_95pc, states = agent.learn_and_aggregate(config)

    return rewards, rewards_95pc, states

if __name__ == "__main__":
    results = {
        "MetaPRL" : benchmark(MetaPRLAgent),
        "PRL" : benchmark(PRLAgent),
        # "RL" : benchmark(RLAgent)
    }
    plot_results(results, "stochastic_transition_results", optimal_reward=-15.)
    # plot_states_heatmap(results)
    for agent, (rewards, rewards_95pc, states) in results.items():
        if agent in ["MetaPRL", "PRL", "DumbPRL"]:
            print(f"{agent} min, max, mean, final planning, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[mb_learn_config_dict['planning_steps']-mb_learn_config_dict['window_size']], rewards[-1]}")
        else:
            print(f"{agent} min, max, mean, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[-1]}")
    