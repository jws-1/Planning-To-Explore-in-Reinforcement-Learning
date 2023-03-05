
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
        T = {(i,j) : {a : [0,0] for a in Action} for i in range(self.world.dim[0]) for j in range(self.world.dim[1])}
        observed = np.full((*self.world.dim, 4), False, dtype=bool)
        observations = np.full(self.world.dim, False, dtype=bool)
        for i in range(config.episodes):
            if i % 100 == 0:
                print(f"RL-A*-AGENT: episode {i}")
            print(i)
            self.world.sample()
            T_episodic = {(i,j) : {a : 0 for a in Action} for i in range(self.world.dim[0]) for j in range(self.world.dim[1])}
            # if i < config.static_threshold:
            #     self.world.sample()
            # else:
            #     self.world.static()

            done = False
            state = self.world.start
            # states[i][state]+=1
            
            planning = i < config.planning_steps

            while not done:
                if planning:
                    if config.learn_model:
                        for s in T.keys():
                            for a in T[s].keys():
                                if observed[s][a.value]:
                                    self.model.T[s][a] = (self.model.T[s][a][0], T[s][a][0] / float(T[s][a][1]))
                        for s in T_episodic.keys():
                            for a in T_episodic[s].keys():
                                if T_episodic[s][a] > 0:
                                    self.model.T[s][a] = (self.model.T[s][a][0], 0.0)
                    if random.uniform(0, 1) < config.eps:
                        action = Action(random.randint(0, 3))
                    else:
                        self.model.start = state
                        action = self.planner.plan(observations)
                else:
                    action = Action(int(np.argmax(self.q_table[state])))
                
                reward, next_state = self.world.action(state, action)
                old_value = self.q_table[state[0]][state[1]][action.value]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                self.q_table[state[0]][state[1]][action.value] = new_value

                if config.learn_model and planning:
                    expected_state = tuple(map(operator.add, state, ACTION_MODIFIERS[action]))
                    if state != next_state and next_state == expected_state:
                        T[state][action][0]+=1
                        if self.model.R[next_state] != reward:
                            self.model.R[next_state] = reward
                    else:
                        # print(state, action, reward, next_state, expected_state)
                        T_episodic[state][action] = 1
                    T[state][action][1]+=1
                    if self.model.feasible_state(expected_state):
                        observations[expected_state] = True
                    observed[state][action.value] = True
                states[i][state]+=1
                state = next_state
                done = state == self.world.goal
                rewards[i] += reward
            states[i][state]+=1
        from pprint import pprint
        pprint(self.model.T)
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

    from mdp import make_reward_function, make_transition_function

    dim = (10,10)
    start = (0,0)
    goal = (9,9)

    obstacles = [((0,3), 0.001), ((1,3), 0.7), ((2,3), 0.5), ((3,3), 1.0), ((4,3), 1.0), ((5,3), 0.65), ((6,3), 0.245), ((7,3), 0.1), ((8,3), 1.0), ((9,3), 0.9)]
    highways = [(0,3), (0,4)]
    T = make_transition_function(dim, obstacles)
    # print(T)
    R = make_reward_function(dim, highways)
    world = GridWorld(dim, start, goal, T, R)
    model = GridWorld(dim, start, goal, make_transition_function(dim, []), make_reward_function(dim, []))


    # world_rewards = np.full(dim, -2, dtype=int)
    
    # world_obstacles = {
    #     (0,3) : 0.,
    #     (2,3) : 0.5,
    #     (3,3) : 1.0,
    #     (4,3) : 1.0,
    #     (5,3) : 0.001
    # }
    # grid = GridWorld(dim, start, goal, world_obstacles, world_rewards)
    agent = ASRLAgent(world, model)
    config = {
        "episodes": 1000,
        "m":1,
        "eps": 0.5,
        "lr":0.1,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps":750,
        "learn_model" :True,
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="rl", save=True, obstacles=obstacles, highways=highways)