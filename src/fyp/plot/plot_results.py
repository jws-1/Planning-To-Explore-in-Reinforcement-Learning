import os
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_reward(agent, rewards, rewards_95pc):
    plt.plot(rewards, label=agent)
    plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)

def plot_states(agent, states, alpha=1.0):
    plt.bar(np.arange(len(states)), states, alpha=alpha, label=agent)

def plot_rewards_alone(agent, rewards, rewards_95pc, dir):
    plt.figure(0)
    plot_reward(agent, rewards, rewards_95pc)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent} Reward/Episode")    
    plt.savefig(os.path.join(dir, f"{agent}-rewards.png"))
    plt.clf()

def plot_states_alone(agent, states, dir):
    plt.figure(0)
    plot_states(agent, states)
    plt.xlabel("State")
    plt.ylabel("Visited")
    plt.title(f"{agent} States")    
    plt.savefig(os.path.join(dir, f"{agent}-states.png"))
    plt.clf()

def plot_results(results, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

    for agent, result in results.items():
        rewards, rewards_95pc, states = result
        print(states.shape)
        states = np.sum(np.sum(states, axis=0), axis=0)
        plot_rewards_alone(agent, rewards, rewards_95pc, dir)
        plot_states_alone(agent, states, dir)

    plt.figure(0)
    for agent, result in results.items():
        rewards, rewards_95pc, states = result
        states = np.sum(np.sum(states, axis=0), axis=0)
        plot_reward(agent, rewards, rewards_95pc)        
    plt.legend()
    plt.savefig(os.path.join(dir, "reward-comparison.png"))
    plt.clf()
    plt.figure(0)
    for agent, result in results.items():
        rewards, rewards_95pc, states = result
        states = np.sum(np.sum(states, axis=0), axis=0)
        plot_states(agent, states, alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(dir, "states-comparison.png"))
    plt.clf()