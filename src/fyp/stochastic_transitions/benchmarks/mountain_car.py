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
    "lr": 0.6,
    "df": 1.0,
    "learn_model":True,
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
    def __init__(self, min_position=-1.2, max_position=0.6, min_velocity=-0.07, max_velocity=0.07, goal_position=0.5):
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space
        self.nA = self.env.action_space.n

        self.nP = 100
        self.nV = 100

        nS = (self.env.observation_space.high - self.env.observation_space.low)*np.array([self.nP, self.nV])
        self.nSP, self.nsV = np.round(nS, 0).astype(int) + 1
        self.nS = self.nSP * self.nsV


        self.min_position = min_position
        self.max_position = max_position
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.goal_position = goal_position
        
    def discretize(self, obs):
        state = (obs - self.env.observation_space.low)*np.array([self.nP, self.nV])
        pos, vel =  np.round(state, 0).astype(int) + 1
        s  = pos * self.nsV + vel
        return min(s, self.nS-1)

    def from_flat_index(self, flat_index):
        pos = flat_index // self.nV
        vel = flat_index % self.nV
        return pos, vel

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
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
    goal_positions = np.arange(0.5, env.max_position+0.1, 0.1)
    goal_states = np.array([env.discretize((pos, 0)) for pos in goal_positions])
    print(goal_states)
    transition_function = np.zeros((env.nS, env.nA, env.nS))
    reward_function = np.full((env.nS, env.nA, env.nS), 0.0)

    for s in range(env.nS):
        for a in range(env.nA):
            for next_s in range(env.nS):
                reward = -1.0
                if next_s in goal_states:
                    reward = 0.0
                reward_function[s, a, next_s] = reward

    # print(reward_function)
    for s in range(env.nS):
        for a in range(env.nA):
            env.reset()
            env.env.state = np.array(env.from_flat_index(s))
            next_s, _, _, _ = env.step(a)
            transition_function[s, a, next_s] = 1.0
    # row_sums = np.sum(transition_function, axis=2)
    # print(row_sums)
    return MDP(np.array(range(env.nS)), goal_states, np.array(range(env.nA)), transition_function, reward_function)

def benchmark(agent_cls, learn=True):
    np.random.seed(42)
    # Create an instance of the Mountain Car environment
    env = MountainCarDiscretized()

    if agent_cls in [MetaPRLAgent, PRLAgent]:
        inaccurate_mdp = create_mdp(env)
        agent = agent_cls(env, inaccurate_mdp)
        env.reset()
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
        # "MetaPRL" : benchmark(MetaPRLAgent),
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
    