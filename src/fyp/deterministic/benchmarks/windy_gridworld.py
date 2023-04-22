from .benchmark_env import BenchmarkEnv
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import gym_windy_gridworlds


class BenchmarkCliffWalking(BenchmarkEnv):

    def __init__(self, seed=42):
        self.env_name = "WindyGridWorld-v0"
        self.seed = seed

    def generate_model(self, reasonable_meta=False, noise=False):
        """
        Generates an inaccurate model for the MB agents.
        The inaccuracy is that the wind is not represented.
        """
        pass

    def generate_reasonable_meta(self):
        pass

    def handle_results(self, results):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "deterministic")):
            os.mkdir(os.path.join("results", "deterministic"))
        if not os.path.exists(os.path.join("results", "deterministic", "windy_gridworld")):
            os.mkdir(os.path.join("results", "deterministic", "windy_gridworld"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states = result
            plt.figure(0)
            plt.plot(rewards, label=agent)
            plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Windy Gridworld {agent} Learning Curve")    
            plt.savefig((os.path.join("results", "deterministic", "windy_gridworld", f"{agent}-rewards.png")))
            plt.clf()

            plt.figure(1)
            plt.plot(rewards, label=agent)
            plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        
        plt.figure(1)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Windy Gridworld")  
        plt.savefig((os.path.join("results", "deterministic", "windy_gridworld", "rewards.png")))
        plt.clf()