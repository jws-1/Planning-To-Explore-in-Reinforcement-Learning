#!/usr/bin/env python3
from mdp import MDP
from meta_agent import RLMetaAgent
from actions import Action
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

env = gym.make("FrozenLake-v1", is_slippery=True, desc=generate_random_map(size=4))
print(env.action_space)

states = list(range(env.nS))
actions = list(range(env.nA))
start = env.reset()
print(start)
print(env.env.P)
import random
env.g = env.nS-1
T_MDP = {state : {a: [] for a in actions } for state in states}

for state in states:
    for action in actions:
        for prob, next_state, reward, done in env.env.P[state][action]:
            # if random.random() < 0.5:
            #     prob = random.uniform(0, 1)
            T_MDP[state][action].append((prob, next_state))

# T_MDP[start] ={
#     0: [(1.0, 0, 0.0, False)],
#     1: [(1.0, 4, 0.0, False)],
#     2: [(1.0, 1, 0.0, False)],
#     3: [ (1.0, 0, 0.0, False)]
# }
R = {state: 0 for state in states}
R[env.nS-1] = 1
mdp = MDP(states, actions, T_MDP, R, 1.0)

agent = RLMetaAgent(env, mdp)

config = {
    "episodes": 1000,
    "m":1,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "window_size":20,
    "planning_steps":100,
    "render":False
}

from types import SimpleNamespace
import numpy as np
rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

min_, max_ = 0, 1000
print(states.shape)
agent.plot_results(rewards[min_:max_], states[:, min_:max_, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True)
print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")


# env.reset()
# print(env.observation_space.high)
#     # env.render()


