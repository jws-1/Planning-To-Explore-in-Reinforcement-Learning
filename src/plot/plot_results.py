import os
import numpy as np
import matplotlib.pyplot as plt

def plot_reward(agent, rewards, rewards_95pc):
    plt.plot(rewards, label=agent)
    plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

def plot_rewards_alone(agent, rewards, rewards_95pc, dir):
    plt.figure(0)
    plot_reward(agent, rewards, rewards_95pc)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent} Reward/Episode")    
    plt.savefig(os.path.join(dir, f"{agent}-rewards.png"))
    plt.clf()

def plot_results(results, dir):

    for agent, result in results.items():
        plot_rewards_alone(agent, *result, dir)

    plt.figure(0)
    for agent, result in results.items():
        plot_reward(agent, *result)
    plt.savefig(os.path.join(dir, "comparison.png"))
    plt.clf()
