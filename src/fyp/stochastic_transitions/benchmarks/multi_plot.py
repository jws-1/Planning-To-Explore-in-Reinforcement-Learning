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


        agent, color = labels[agent]
        if env in ["frozen_lake"]:

            print(f"FROZEN: {np.cumsum(rewards)[-1]}")
            axs[i].plot(np.cumsum(rewards), label=agent, color=color)
        else:
            axs[i].plot(rewards, label=agent, color=color)

            axs[i].fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2, color=color)

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



    axs[i].set_title(f"{prettify_name(env)}")
    axs[i].set_title(f"{prettify_name(env)}")
    axs[i].legend()
    axs[i].legend()

fig.text(0.5, 0.04, 'Episode', ha='center')
fig.text(0.04, 0.5, 'Reward', va='center', rotation='vertical')
plt.show()
