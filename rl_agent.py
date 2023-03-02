
import numpy as np
import random
from actions import Action
from gridworld import GridWorld
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os
from copy import deepcopy
from types import SimpleNamespace

class RLAgent():

    def __init__(self, world):
        np.random.seed(42)
        self.world = world
        self.q_table = np.full((*self.world.dim, 4), 100000., dtype=float)

    def reset(self):
        self.q_table = np.full((*self.world.dim, 4), 100000., dtype=float)

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, *self.world.dim))

        for i in range(config.episodes):
            if i % 100 == 0:
                print(f"RL-AGENT: episode {i}")

            if i < config.static_threshold:
                self.world.sample()
            else:
                self.world.static()

            done = False
            state = self.world.current

            while not done:

                if random.uniform(0, 1) < config.eps:
                    action = Action(random.randint(0, 3))
                else:
                    action = Action(int(np.argmax(self.q_table[state])))

                reward, done = self.world.action(action)
                next_state = self.world.current

                old_value = self.q_table[state[0]][state[1]][action.value]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                self.q_table[state[0]][state[1]][action.value] = new_value

                states[i][state]+=1
                state = next_state
                rewards[i] += reward
                
            states[i][state]+=1
        return rewards, states

    def learn_and_aggregate(self, config):
        rewards_windows = np.zeros((config.m, config.episodes - config.window_size+1, config.window_size))
        states = np.zeros((config.m, config.episodes, *self.world.dim))
        for i in range(config.m):

            rewards, states_ = self.learn(config)

            rewards_window = np.lib.stride_tricks.sliding_window_view(rewards, config.window_size)
            rewards_windows[i] = deepcopy(rewards_window)
            
            states[i] = states_

        # Calculate 95% confidence interval for each window.

        aggregated_rewards_windows = np.mean(np.mean(rewards_windows, axis=2), axis=0)
        rewards_95pc = []
        for i in range(rewards_windows.shape[1]):
            a = []
            for j in range(rewards_windows.shape[0]):
                a.extend(list(np.ravel(rewards_windows[j][i])))
            rewards_95pc.append(st.norm.interval(confidence=0.95, loc=np.mean(a), scale=st.sem(a)))

        return aggregated_rewards_windows, np.array(rewards_95pc), states

    def plot_results(self, rewards, states, policy="e-greedy", rewards_95pc=None, save=True, config=None):
        if not os.path.exists(policy) and save:
            os.mkdir(policy)
        print(rewards.shape[0]-50)
        plt.figure(0)
        plt.plot(rewards)
        plt.fill_between(np.arange(len(rewards_95pc)), rewards_95pc[:, 0], rewards_95pc[:, 1], alpha=0.2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"RL Agent Reward/Episode {(policy)}")
        if save:
            plt.savefig(os.path.join(policy, f"{policy}-rewards.png"))
            plt.clf()


        cmap = plt.cm.get_cmap("Reds")
        cmap.set_bad("black")
        plt.figure(1)
        annots = [ [""]*self.world.dim[0] for _ in range(self.world.dim[1]) ]
        annots[self.world.start[0]][self.world.start[1]] = "S"
        annots[self.world.goal[0]][self.world.goal[1]] = "G"
        
        for i in range(self.world.dim[0]):
            for j in range(self.world.dim[1]):
                if (i,j)  in self.world.static_obstacles:
                    annots[i][j] = "O"
                else:
                    annots[i][j] += " " + str(self.world.static_rewards[i,j])
                    annots[i][j] = annots[i][j].lstrip()

        annots = np.array(annots)
        if states.ndim == 4:
            states_all = np.sum(np.sum(states, axis=0), axis=0)
        elif states.ndim == 3:
            states_all = np.sum(states, axis=0)
        for obstacle in self.world.static_obstacles:
            states_all[obstacle] = np.nan
        # states_all[self.world.static_obstacles] = np.nan
        hm = sns.heatmap(np.transpose(states_all), linewidth=0.5, annot=np.transpose(annots), square=True, fmt='', cmap=cmap)
        hm.invert_yaxis()
        plt.title(f"RL Agent Visited States ({policy})")
        if save:
            plt.savefig(os.path.join(policy, f"{policy}-states.png"))
            plt.clf()

        plt.figure(2)
        if states.ndim == 4:
            if hasattr(config, "planning_steps"):
                states_exp = np.sum(np.sum(states[:, :config.planning_steps, :, :], axis=0), axis=0)
            else:
                states_exp = np.sum(np.sum(states[:, :50, :, :], axis=0), axis=0)
        elif states.ndim == 3:
            if hasattr(config, "planning_steps"):
                states_exp = np.sum(states[:, :config.planning_steps, :, :], axis=0)
            else:
                states_exp = np.sum(states[:, :50, :, :], axis=0)
        for obstacle in self.world.static_obstacles:
            states_exp[obstacle] = np.nan
        hm = sns.heatmap(np.transpose(states_exp), linewidth=0.5, annot=np.transpose(annots), square=True, fmt='', cmap=cmap)
        hm.invert_yaxis()
        plt.title(f"RL Agent Visited States ({policy}): Initial Exploration")
        if save:
            plt.savefig(os.path.join(policy, f"{policy}-initial-exploration.png"))
            plt.clf()

        plt.figure(3)
        if states.ndim == 4:
            states_fin = np.sum(np.sum(states[:, rewards.shape[0]-50:, :, :], axis=0), axis=0)
        elif states.ndim == 3:
            states_fin = np.sum(states[:, rewards.shape[0]-50:, :, :], axis=0)
        for obstacle in self.world.static_obstacles:
            states_fin[obstacle] = np.nan
        hm = sns.heatmap(np.transpose(states_fin), linewidth=0.5, annot=np.transpose(annots), square=True, fmt='', cmap=cmap)
        hm.invert_yaxis()
        plt.title(f"RL Agent Visited States ({policy}): Final Policy")
        if save:
            plt.savefig(os.path.join(policy, f"{policy}-final-policy.png"))
            plt.clf()
        else:
            plt.show()

if __name__ == "__main__":
    dim = (10,10)
    start = (0,0)
    goal = (9,9)

    world_rewards = np.full(dim, -2, dtype=int)
    
    world_obstacles = {
        (0,3) : 0.,
        (2,3) : 0.5,
        (3,3) : 1.0,
        (4,3) : 1.0,
        (5,3) : 0.001
    }
    grid = GridWorld(dim, start, goal, world_obstacles, world_rewards)
    agent = RLAgent(grid)
    config = {
        "episodes": 1000,
        "m":20,
        "eps": 0.1,
        "lr":0.1,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="rl", save=True)