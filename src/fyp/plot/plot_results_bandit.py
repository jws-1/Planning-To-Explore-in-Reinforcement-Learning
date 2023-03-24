import os
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_reward(agent, rewards, rewards_95pc):
    plt.plot(rewards, label=agent)
    plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

def plot_arms_alone(agent, arms, dir):
    plt.figure(0)
    arms = np.sum(np.sum(arms, axis=0), axis=0)
    plt.bar(np.arange(len(arms)), arms, alpha=1.0, label=agent)
    plt.xlabel("Arms")
    plt.ylabel("Pulled")
    plt.title(f"{agent} Arms")    
    plt.savefig(os.path.join(dir, f"{agent}-arms.png"))
    plt.clf()

def plot_rewards_alone(agent, rewards, rewards_95pc, dir):
    plt.figure(0)
    plot_reward(agent, rewards, rewards_95pc)
    plt.xlabel("Action")
    plt.ylabel("Reward")
    plt.title(f"{agent} Reward/Action")    
    plt.savefig(os.path.join(dir, f"{agent}-rewards.png"))
    plt.clf()


def plot_result_bandit(results, dir, optimal_reward=None):
    if not os.path.exists(dir):
        os.mkdir(dir)

    for agent, result in results.items():
        rewards, rewards_95pc, arms = result
        plot_rewards_alone(agent, rewards, rewards_95pc, dir)
        plot_arms_alone(agent, arms, dir)

    plt.figure(0)
    for agent, result in results.items():
        rewards, rewards_95pc, arms = result
        plot_reward(agent, rewards, rewards_95pc)

    if optimal_reward is not None:
        plt.hlines(optimal_reward, plt.axis()[1], 1., linestyles="dashed", alpha=0.2, label="Optimal")
    
    plt.legend()
    plt.savefig(os.path.join(dir, "reward-comparison.png"))
    plt.clf()