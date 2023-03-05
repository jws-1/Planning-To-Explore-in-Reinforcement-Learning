import matplotlib.pyplot as plt
from meta_agent import RLMetaAgent
from rl_astar_agent import ASRLAgent
from rl_agent import RLAgent
from types import SimpleNamespace
from gridworld import GridWorld
import numpy as np

from mdp import make_reward_function, make_transition_function

obstacles = [
    ((2,1), 1.0),
    ((2,2), 1.0),
    ((2,3), 1.0),
    ((2,4), 1.0),
    ((2,5), 1.0),
    ((2,6), 1.0),
    ((2,7), 1.0),
    ((3,7), 0.5),
    ((4,7), 1.0),
    ((5,7), 1.0),
    ((6,7), 1.0),
    ((7,1), 1.0),
    ((7,2), 1.0),
    ((7,3), 1.0),
    ((7,4), 1.0),
    ((7,5), 1.0),
    ((7,6), 1.0),
    ((7,7), 1.0),
]

inaccurate_ = [
    ((2,1), 1.0),
    ((2,2), 1.0),
    ((2,3), 1.0),
    ((2,4), 1.0),
    ((2,5), 1.0),
    ((2,6), 1.0),
    ((2,7), 1.0),
    ((3,7), 1.0),
    ((4,7), 0.0),
    ((5,7), 0.0),
    ((6,7), 1.0),
    ((7,1), 1.0),
    ((7,2), 1.0),
    ((7,3), 1.0),
    ((7,4), 1.0),
    ((7,5), 1.0),
    ((7,6), 1.0),
    ((7,7), 1.0),
]
highways = [
    (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8),
    (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8),
    (7,8), (6,8), (5,8), (4,8), (3,5), (3,6)
]

def setup(optimistic=False, inaccurate=True):


    dim = (10,10)
    start = (4,0)
    goal = (3,8)


    T = make_transition_function(dim, obstacles)
    # print(T)
    R = make_reward_function(dim, highways)

    world = GridWorld(dim, start, goal, T, R)

    if optimistic:
        model = GridWorld(dim, start, goal, make_transition_function(dim, []), make_reward_function(dim, [], -1., -1.))
    elif inaccurate:
        model = GridWorld(dim, start, goal, make_transition_function(dim, inaccurate_), make_reward_function(dim, []))
    else:
        model = GridWorld(dim, start, goal, make_transition_function(dim, []), make_reward_function(dim, []))

    return world, model

min_, max_ = 0,1000

a_star_learn_agent = ASRLAgent(*setup(False))

config = {
    "episodes": 150,
    "m":20,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "eps" : 0.5,
    "window_size":20,
    "learn_model" : True,
    "planning_steps":50,
}

a_star_learn_rewards, a_star_learn_rewards95pc, a_star_learn_states = a_star_learn_agent.learn_and_aggregate(SimpleNamespace(**config))
a_star_learn_agent.plot_results(a_star_learn_rewards, a_star_learn_states, "a-star-learn", a_star_learn_rewards95pc, config=config, obstacles=obstacles, highways=highways)

a_star_learn_optimistic_agent = ASRLAgent(*setup(True))

a_star_learn_optimistic_rewards, a_star_learn_optimistic_rewards95pc, a_star_learn_optimistic_states = a_star_learn_optimistic_agent.learn_and_aggregate(SimpleNamespace(**config))
a_star_learn_optimistic_agent.plot_results(a_star_learn_optimistic_rewards, a_star_learn_optimistic_states, "a-star-learn-optimistic", a_star_learn_optimistic_rewards95pc, config=config, obstacles=obstacles, highways=highways)

# a_star_agent = ASRLAgent(*setup())

# config = {
#     "episodes": 150,
#     "m":1,
#     "lr":0.6,
#     "df":1.0, # episodic, so rewards are undiscounted.
#     "eps" : 0.5,
#     "window_size":20,
#     "learn_model" : True,
#     "static_threshold": 25,
#     "planning_steps":200,
#     "memory_episodes":5
# }

# a_star_rewards, a_star_rewards_95pc, a_star_states = a_star_agent.learn_and_aggregate(SimpleNamespace(**config))
# a_star_agent.plot_results(a_star_rewards, a_star_states, "a-star", a_star_rewards_95pc)


config = {
    "episodes": 150,
    "m":20,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "eps" : 0.5,
    "window_size":20,
}

rl_agent = RLAgent(setup()[0])
rewards, rewards_95pc, states = rl_agent.learn_and_aggregate(SimpleNamespace(**config))
rl_agent.plot_results(rewards, states, "e-greedy", rewards_95pc, config=config, obstacles=obstacles, highways=highways)


meta_agent = RLMetaAgent(*setup(False, True))

config = {
    "episodes": 150,
    "m":20,
    "lr":0.6,
    "df":1.0, # episodic, so rewards are undiscounted.
    "window_size":20,
    "planning_steps" : 50,
    "learn_model" : True,
}
meta_rewards, meta_rewards_95pc, meta_states = meta_agent.learn_and_aggregate(SimpleNamespace(**config))

meta_agent.plot_results(meta_rewards[min_:max_], meta_states[:, min_:max_, :, :], rewards_95pc=meta_rewards_95pc[min_:max_,:], policy="meta", save=True, config=config, obstacles=obstacles, highways=highways)


# plt.figure(0)
# plt.plot(meta_rewards, label=fr"RL-Meta ($N_\mu=30$)")
# plt.fill_between(np.arange(len(meta_rewards_95pc)), meta_rewards_95pc[:, 0], meta_rewards_95pc[:, 1], alpha=0.2)
# for i in range(1, 6):
#     meta_agent_ = RLMetaAgent(*setup())
# # meta_agent_5 = RLMetaAgent(*setup())

#     config = {
#         "episodes": 500,
#         "m":20,
#         "lr":0.6,
#         "df":1.0, # episodic, so rewards are undiscounted.
#         "window_size":20,
#         "planning_steps" : 120,
#         "learn_model" : True,
#         "memory_episodes": i * 5,
#         "static_threshold": 100 
#     }
#     meta_rewards_, meta_rewards_95pc_, meta_states_ = meta_agent_.learn_and_aggregate(SimpleNamespace(**config))
#     # meta_agent_.plot_results(meta_rewards_[min_:max_], meta_states_[:, min_:max_, :, :], rewards_95pc=meta_rewards_95pc_[min_:max_,:], policy=f"meta_{i*5}", save=True, config=config)
#     plt.plot(meta_rewards_, label=fr"RL-Meta ($N_\mu={i*5}$)")
#     plt.fill_between(np.arange(len(meta_rewards_95pc_)), meta_rewards_95pc_[:, 0], meta_rewards_95pc_[:, 1], alpha=0.2)

# plt.vlines(100, plt.axis()[2], 1., linestyle="dashed", alpha=0.2, label="Planning steps end")
# plt.vlines(80, plt.axis()[2], 1., linestyle="dashed", alpha=0.2, label="Deterministic environment", colors=['red'])

# plt.legend()
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title(fr"Reward/Episode. $\alpha=0.6$, $\gamma=1.0$")
# plt.savefig("comparison/meta_rewards.png")

plt.figure(1)
plt.plot(meta_rewards, label=fr"RL-Meta")
plt.fill_between(np.arange(len(meta_rewards_95pc)), meta_rewards_95pc[:, 0], meta_rewards_95pc[:, 1], alpha=0.2)
plt.plot(rewards, label=fr"$\epsilon$-greedy,$\epsilon=0.5$")
plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
plt.plot(a_star_learn_rewards, label=fr"RL-A* (model-learning,$\epsilon=0.5$)")
plt.fill_between(np.arange(len(a_star_learn_rewards95pc)), a_star_learn_rewards95pc[:, 0], a_star_learn_rewards95pc[:, 1], alpha=0.2)
# plt.plot(a_star_rewards, label="a_star")
# plt.fill_between(np.arange(len(a_star_rewards_95pc)), a_star_rewards_95pc[:, 0], a_star_rewards_95pc[:, 1], alpha=0.2)
plt.plot(a_star_learn_optimistic_rewards, label=fr"RL-A* (model-learning, optimistic, $\epsilon=0.5$)")
plt.fill_between(np.arange(len(a_star_learn_optimistic_rewards95pc)), a_star_learn_optimistic_rewards95pc[:, 0], a_star_learn_optimistic_rewards95pc[:, 1], alpha=0.2)


plt.vlines(10, plt.axis()[2], 1., linestyle="dashed", alpha=0.2, label="Planning steps end")
# plt.vlines(80, plt.axis()[2], 1., linestyle="dashed", alpha=0.2, label="Deterministic environment", colors=['red'])

plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(fr"Reward/Episode. $\alpha=0.6$, $\gamma=1.0$")
plt.savefig("comparison/rewards.png")


print(meta_rewards)
print(f"Meta mean, max, final planning, final model-free reward: {np.mean(meta_rewards), np.max(meta_rewards), meta_rewards[config['planning_steps']], meta_rewards[-1]}")
# print(f"RL-A* mean, max, final planning, final model-free reward: {np.mean(a_star_rewards), np.max(a_star_rewards), a_star_rewards[config['planning_steps']-config['window_size']], a_star_rewards[-1]}")
print(f"RL-A* (learn) mean, max, final planning, final model-free reward: {np.mean(a_star_learn_rewards), np.max(a_star_learn_rewards), a_star_learn_rewards[config['planning_steps']], a_star_learn_rewards[-1]}")
print(f"RL-A* (learn, optimistic) mean, max, final planning, final model-free reward: {np.mean(a_star_learn_optimistic_rewards), np.max(a_star_learn_optimistic_rewards), a_star_learn_optimistic_rewards[config['planning_steps']], a_star_learn_optimistic_rewards[-1]}")
print(f"RL mean, max, final planning, final model-free reward: {np.mean(rewards), np.max(rewards), rewards[config['planning_steps']], rewards[-1]}")
