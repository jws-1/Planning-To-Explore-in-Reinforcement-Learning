
import numpy as np
import random
from actions import Action, ACTION_MODIFIERS
from gridworld import GridWorld
import matplotlib.pyplot as plot
import seaborn as sns
from rl_agent import RLAgent
from math import exp
from copy import deepcopy
from planner import Planner
from types import SimpleNamespace
import operator


class ASRLAgent(RLAgent):

    def __init__(self, world, model):
        np.random.seed(42)
        self.world = world
        self.initial_model = deepcopy(model)
        self.model = model
        self.q_table = np.full((*self.world.dim, 4), 0, dtype=float)

    def reset(self):
        self.q_table = np.full((*self.world.dim, 4), 0, dtype=float)
        self.model = deepcopy(self.initial_model)
        self.planner = Planner(self.model)

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, *self.world.dim))
        observed_obstacles = np.zeros((config.episodes, *self.world.dim))

        for i in range(config.episodes):
            if i % 100 == 0:
                print(f"RL-A*-AGENT: episode {i}")

            if i < config.static_threshold:
                self.world.sample()
            else:
                self.world.static()

            done = False
            state = self.world.current
            # states[i][state]+=1
            
            planning = i < config.planning_steps

            while not done:
                self.model.current = deepcopy(self.world.current)
                if planning:
                    if i == 0:
                        P = observed_obstacles[i]
                    else:
                        P = np.mean(observed_obstacles[max(0, i-config.memory_episodes):i], axis=0)
                    self.model.obstacles = {}
                    for k in range(P.shape[0]):
                        for j in range(P.shape[1]):
                            if P[k,j] >= 0.5:
                                self.model.obstacles[(k,j)] = P[k,j]
                            if observed_obstacles[i][k][j] > 0.0:
                                self.model.obstacles[(k,j)] = 1.0
                if planning:
                    if random.uniform(0, 1) < config.eps:
                        action = Action(random.randint(0, 3))
                    else:
                        action = self.planner.plan()
                else:
                    action = Action(int(np.argmax(self.q_table[state])))
                
                reward, done = self.world.action(action)
                next_state = self.world.current
                old_value = self.q_table[state[0]][state[1]][action.value]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                self.q_table[state[0]][state[1]][action.value] = new_value

                if config.learn_model and planning:
                    expected_state = tuple(map(operator.add, state, ACTION_MODIFIERS[action]))
                    if self.model.ok_state(expected_state, episodic=False):
                        expected_reward = self.model.rewards[expected_state]
                        if self.world.current != expected_state:
                            observed_obstacles[i][expected_state[0]][expected_state[1]] = 1.0
                        elif expected_reward != reward:
                            self.model.rewards[expected_state] = reward
                
                states[i][state]+=1
                state = next_state

                rewards[i] += reward
            states[i][state]+=1
        return rewards, states

if __name__ == "__main__":
    # N = 10memory_episodes
    # start = (0,0)
    # goal = (N-1, N-1)


    # model_rewards = np.full((N, N), -2, dtype=int)
    # world_rewards = np.full((N, N), -2, dtype=int)


    # model_rewards[goal] = 0
    # world_rewards[goal] = 0

    # world_highway = [(0,4), (1,9), (2,9), (5,9), (6,9)]

    # model = np.full((N,N), False, dtype=bool)
    # model_obstacles = [(0,4), (0,5), (9,0)]

    # for obstacle in model_obstacles:
    #     model[obstacle] = True
    #     model_rewards[obstacle] = -10

    # world = np.full((N,N), False, dtype=bool)
    # world_obstacles = [(7,9), (9,8)]

    # for obstacle in world_obstacles:
    #     world[obstacle] = True
    #     world_rewards[obstacle] = -10

    # for highway in world_highway:
    #     world_rewards[highway] = -1

    # grid = GridWorld(N, start, goal, model_obstacles, world_obstacles, model_rewards, world_rewards)
    # agent = ASRLAgent(grid)
    # epochs, epochs_95pc, rewards, rewards_95pc, states = agent.learn_and_aggregate(lr=0.3, eps=0.5, n=1000, df=0.9, m=20, window_size=20)

    # agent.plot_results(epochs, rewards, states, epochs_95pc=epochs_95pc, rewards_95pc=rewards_95pc, policy="asrl", save=True)

    dim = (10,10)
    start = (0,0)
    goal = (9,9)

    world_rewards = np.full(dim, -2, dtype=int)
    
    world_obstacles = {
        (0,3) : 0.5,
        (1,3) : 0.1,
        (2,3) : 0.5,
        (3,3) : 1.0,
        (4,3) : 1.0,
        (5,3) : 0.001,
        (6,3) : 0.5,
        (7,3) : 0.3,
        (8,3) : 0.5,
        (9,3) : 1.0
    }

    static_obstacles = [
        (0,3), (1,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3)
    ]

    # posterior_world_obstacles = {
    #     (0,3) : 0.5,
    #     (1,3) : 0.1,
    #     (2,3) : 0.5,
    #     (3,3) : 1.0,
    #     (4,3) : 1.0,
    #     (5,3) : 0.001,
    #     (6,3) : 0.01,
    #     (7,3) : 0.3,
    #     (8,3) : 0.5,
    #     (9,3) : 1.0
    # }

    world = GridWorld(dim, start, goal, world_obstacles, world_rewards, static_obstacles=static_obstacles, static_rewards=world_rewards)

    model = GridWorld(dim, start, goal, {}, world_rewards)

    agent = ASRLAgent(world, model)
    config = {
        "episodes": 1000,
        "m":1,
        "eps": 0.1,
        "lr":0.1,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps" : 200,
        "learn_model" : True,
        "memory_episodes": 200,
        "static_threshold": 100
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="a_star", save=True)
