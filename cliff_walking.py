#!/usr/bin/env python3
from mdp import MDP
from meta_agent import RLMetaAgent
from rl_agent import RLAgent
from actions import Action
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

env =env = gym.make('CliffWalking-v0')

print(env.action_space)

states = list(range(env.nS))
actions = list(range(env.nA))
state = env.reset()
print(state)
print(env.env.P)
pass
import random
env.g = 47
T_MDP = {state : {a: [] for a in actions } for state in states}

for state in states:
    for action in actions:
        for prob, next_state, reward, done in env.env.P[state][action]:
            if random.random() < 0.5:
                prob = random.uniform(0, 1)
            T_MDP[state][action].append((prob, next_state))
        P = sum([t[0] for t in T_MDP[state][action]])
        T_MDP[state][action] = [(t[0] / P, t[1]) for t in T_MDP[state][action]]

T_MDP[36][0] = [(0.8, 24), (0.2, 36)]

print(T_MDP)

R = {state: -2 for state in states}

mdp = MDP(states, actions, T_MDP, R, 1.0)

meta_agent = RLMetaAgent(env, mdp)

meta_config = {
    "episodes": 100,
    "m":1,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "window_size":5,
    "planning_steps":20,
    "render":False
}

from types import SimpleNamespace
import numpy as np
meta_rewards, meta_rewards_95pc, meta_states = meta_agent.learn_and_aggregate(SimpleNamespace(**meta_config))

min_, max_ = 0, 1000
# print(meta_states.shape)
meta_agent.plot_results(meta_rewards[min_:max_], meta_states[:, min_:max_, :], rewards_95pc=meta_rewards_95pc[min_:max_,:], policy="meta", save=True)

rl_agent = RLAgent(env)

config = {
    "episodes": 100,
    "m":1,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "window_size":5,
    "render":False,
    "eps": 0.1,
}

rewards, rewards_95pc, states = rl_agent.learn_and_aggregate(SimpleNamespace(**config))
rl_agent.plot_results(rewards[min_:max_], states[:, min_:max_, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="rl", save=True)
print(f"RL low, mean, high, final rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[-1]}")
print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(meta_rewards), np.mean(meta_rewards), np.max(meta_rewards), meta_rewards[meta_config['planning_steps']-meta_config['window_size']], meta_rewards[-1]}")

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(meta_rewards, label=fr"RL-Meta")
plt.fill_between(np.arange(len(meta_rewards_95pc)), meta_rewards_95pc[:, 0], meta_rewards_95pc[:, 1], alpha=0.2)
plt.plot(rewards, label=fr"$\epsilon$-greedy,$\epsilon=0.5$")
plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(fr"Reward/Episode. $\alpha=0.6$, $\gamma=1.0$")
plt.savefig("comparison/rewards.png")

# env.reset()
# print(env.observation_space.high)
#     # env.render()


