import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


labels = {
    "RL_VI_Meta_reasonable" : ("RL VI Meta", "#d62728ff"),
    "PRL" : ("PRL", "#1f77b4ff"),
    "RL" : ("RL", "#2ca02cff"),
    "MetaPRL_learn" : ("RL VI Meta (Learn)", "#9467bdff"),
}

def prettify_name(name):
    words = name.split("_")
    words = [word.capitalize() for word in words]

    return " ".join(words)

results_path = os.path.join("results", "stochastic")
envs = os.listdir(results_path)
results_dict = {}
optimal_paths = {}
for env in envs:
    results = os.listdir(os.path.join(results_path, env))
    agents = list(set([f.split("-")[0] for f in results if f.endswith(".npy")]))
    results_dict[env] = {agent : {} for agent in agents}
    for agent in agents:
        results_dict[env][agent]["rewards"] = np.load(os.path.join(results_path, env, f"{agent}-rewards.npy"))
        results_dict[env][agent]["rewards_95pc"] = np.load(os.path.join(results_path, env, f"{agent}-rewards_95pc.npy"))
        results_dict[env][agent]["states"] = np.load(os.path.join(results_path, env, f"{agent}-states.npy"))
        results_dict[env][agent]["states_95pc"] = np.load(os.path.join(results_path, env, f"{agent}-states_95pc.npy"))
        results_dict[env][agent]["regrets"] = np.load(os.path.join(results_path, env, f"{agent}-regrets.npy"))
        results_dict[env][agent]["regrets_95pc"] = np.load(os.path.join(results_path, env, f"{agent}-regrets_95pc.npy"))


# optimal_paths["frozen_lake"] = [(36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47), -13.0]
# optimal_paths["gridworld"] = [(95, 85, 75, 65, 55), -4.0]
# optimal_paths["windy_gridworld"] = [(30, 31, 32, 33, 24, 15, 6, 7, 8, 9, 19, 29, 39, 49, 48, 37), -15.0]
# print(np.mean(results_dict["gridworld"]["MetaPRL_reasonable"]["states"], axis=0).shape)

# optimal_path = [95, 85, 75, 65, 55]
# sub_optimal_states = [0 for _ in range(200)]
# for episode, states in enumerate(results_dict["gridworld"]["MetaPRL_reasonable"]["states"].tolist()):
#     for state, count in enumerate(states):
#         if state not in optimal_path:
#             sub_optimal_states[episode]+=count
# plt.plot(sub_optimal_states)
# plt.show()
# env = "gridworld"

# # ax = plt.gca()
# # axins = ax.inset_axes([0.2, 0.2, 0.5, 0.5])
# # ax.indicate_inset_zoom(axins)
# for agent, results in results_dict[env].items():
#     rewards = results["rewards"]
#     rewards_95pc = results["rewards_95pc"]
#     plt.plot(rewards, label=agent)
#     plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
# #     axins.plot(rewards, label=agent)
# #     axins.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
# # axins.set_ylim((min(np.min(results_dict[env]["MetaPRL_reasonable"]["rewards"]), np.min(results_dict[env]["PRL"]["rewards"]))-10, 10))

# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title(f"{prettify_name(env)}")
# plt.legend(loc="lower right")
# plt.show()
print(len(envs))
fig, axs = plt.subplots(len(envs), 1, figsize=(8, 12))

# print(results_dict)

for i in range(len(envs)):
    env = envs[i]
    summary_data = []
    summary_columns = ["Min", "Max", "Mean", "Std. Dev", "Final"]
    row_labels = []

    # create inset axes

    if env in ["cliff_walking"]:
        axins_1 = axs[i].inset_axes([0.2, 0.2, 0.5, 0.5])
        # axins_2 = axs[i][1].inset_axes([0.2, 0.2, 0.5, 0.5])
        axins_1.set_xticklabels([])
        axins_1.set_yticklabels([])
        # axins_2.set_xticklabels([])
        # axins_2.set_yticklabels([])
        axs[i].indicate_inset_zoom(axins_1)
        # axs[i][1].indicate_inset_zoom(axins_2)

    for agent, results in results_dict[env].items():
        print(env, agent)
        # print(np.mean(results["states"], axis=0).shape)
        rewards = results["rewards"]
        rewards_95pc = results["rewards_95pc"]
        states = results["states"].tolist()
        regrets = results["regrets"]
        regrets_95pc = results["regrets_95pc"]
        # states = np.mean(states, axis=0)


        # states = [sum([1 for s in episodic_state if s not in optimal_paths[env][0]]) for episodic_state in states]
        # print(states)

        agent, color = labels[agent]
        if env in ["frozen_lake"]:

            print(f"FROZEN: {np.cumsum(rewards)[-1]}")
            axs[i].plot(np.cumsum(rewards), label=agent, color=color)
            # axs[i].fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0] + np.cumsum(rewards), rewards_95pc[:, 1] + np.cumsum(rewards), alpha=0.2, color=color)
        else:
            axs[i].plot(rewards, label=agent, color=color)

            axs[i].fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2, color=color)
            # axs[i][1].plot(regrets, label=agent)
            # axs[i][1].fill_between(np.arange(len(regrets_95pc)), regrets_95pc[:, 0], regrets_95pc[:, 1], alpha=0.2)
            
        # non_optimal = [0 for _ in range(len(states))]
        # for k, episode in enumerate(states):
        #     for j, c in enumerate(episode):
        #         if j not in optimal_paths[env][0] or c > 0:
        #             non_optimal[k]+=c
        # print(non_optimal)

        # axs[i][2].plot(np.arange(len(non_optimal)), non_optimal, label=agent)

        # axs[i][3].plot(optimal_paths[env][1]-rewards)

        min_, max_, mean, std, final = np.min(rewards), np.max(rewards), np.mean(rewards), np.std(rewards), rewards[-1]
        min_, max_, mean, std, final = np.round(min_, 3), np.round(max_, 3), np.round(mean, 3), np.round(std, 3), np.round(final, 3)
        summary_data.append([min_, max_, mean, std, final])
        print(env, agent, summary_data[-1])
        row_labels.append(agent)

        if env in ["cliff_walking"]:
            # plot zoomed-in area
            axins_1.plot(rewards, label=agent, color=color)
            axins_1.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2, color=color)
            axins_1.set_ylim((min(np.min(results_dict[env]["RL_VI_Meta_Reasonable"]["rewards"]), np.min(results_dict[env]["PRL"]["rewards"]))-10, 10))

            # axins_2.plot(regrets, label=agent)
            # axins_2.fill_between(np.arange(len(regrets_95pc)), regrets_95pc[:, 0], regrets_95pc[:, 1], alpha=0.2)
            # axins_2.set_ylim(-100, 100)

    axs[i].set_title(f"{prettify_name(env)}")
    axs[i].set_title(f"{prettify_name(env)}")
    axs[i].legend()
    axs[i].legend()

fig.text(0.5, 0.04, 'Episode', ha='center')
fig.text(0.04, 0.5, 'Reward', va='center', rotation='vertical')
# fig.text(0.8, 0.05, 'Episode', ha='center')
# fig.text(0.96, 0.5, 'Regret', va='center', rotation='vertical')
plt.show()
