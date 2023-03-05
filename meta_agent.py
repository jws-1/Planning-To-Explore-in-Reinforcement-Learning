from rl_agent import RLAgent
from copy import deepcopy
import numpy as np
from meta_planner import MetaPlanner
from actions import Action, MetaAction, ACTION_MODIFIERS
from gridworld import GridWorld
import operator
from types import SimpleNamespace
from sys import exit
from watchpoints import watch
class RLMetaAgent(RLAgent):

    def __init__(self, world, model):
        self.world = world
        self.initial_model = deepcopy(model)
        self.model = model
        self.q_table = np.full((*self.world.dim, 4), 0., dtype=float)

    def reset(self):
        self.q_table = np.full((*self.world.dim, 4), 0., dtype=float)
        self.model = deepcopy(self.initial_model)
        self.planner = MetaPlanner(self.model)
    
    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, *self.world.dim))
        T = {(i,j) : {a : [0,0] for a in Action} for i in range(self.world.dim[0]) for j in range(self.world.dim[1])}
        observed = np.full((*self.world.dim, 4), False, dtype=bool)
        for i in range(config.episodes):

            T_episodic = {(i,j) : {a : 0 for a in Action} for i in range(self.world.dim[0]) for j in range(self.world.dim[1])}
            episodic_meta_calls = np.full((*self.world.dim, 6), False, dtype=bool)
            episodic_observations = np.full(self.world.dim, False, dtype=bool)


            state = self.world.start

            episodic_meta_calls[state][0] = True
            episodic_meta_calls[state][1] = True
            episodic_meta_calls[state][2] = True
            episodic_meta_calls[state][3] = True
            episodic_meta_calls[state][4] = True
            episodic_meta_calls[state][5] = True
            episodic_observations[state] = True

            if i % 100 == 0:
                print(f"Meta Agent, episode {i}")

            planning = i < config.planning_steps

            self.world.sample()
            
            done = False
            state = self.world.start
            # states[i][state]+=1

            while not done:

                if planning:
                    for s in T.keys():
                        for a in T[s].keys():
                            if observed[s][a.value]:
                                self.model.T[s][a] = (self.model.T[s][a][0], T[s][a][0] / float(T[s][a][1]))
                    for s in T_episodic.keys():
                        for a in T_episodic[s].keys():
                            if T_episodic[s][a] > 0:
                                self.model.T[s][a] = (self.model.T[s][a][0], 0.0)
                    self.model.start = state
                    action, target = self.planner.plan(episodic_meta_calls, episodic_observations)
                else:
                    action = Action(int(np.argmax(self.q_table[state])))

                if isinstance(action, Action):
                    reward, next_state = self.world.action(state, action)
                else:
                    reward, next_state = self.model.meta_action(state, action, target)
                # print(state, action, reward, planning)
                if isinstance(action, Action):
                    old_value = self.q_table[state[0]][state[1]][action.value]
                    next_max = np.max(self.q_table[next_state])
                    new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                    self.q_table[state[0]][state[1]][action.value] = new_value
                    if state != next_state:
                        states[i][state]+=1
                    if planning:
                        expected_state = tuple(map(operator.add, state, ACTION_MODIFIERS[action]))

                        if state != next_state and next_state == expected_state:
                            T[state][action][0]+=1
                            if self.model.R[next_state] != reward:
                                self.model.R[next_state] = reward
                        else:
                            print(state, action)
                            T_episodic[state][action] = 1
                        T[state][action][1]+=1
                        observed[state][action.value] = True
                        episodic_observations[expected_state] = True

                else:
                    episodic_meta_calls[target][action.value] = True
                
                state = next_state
                done = state == self.world.goal
                rewards[i]+=reward
            states[i][state]+=1

        return rewards, states

"""
Agent develops a belief about the state space (transition probabilities < 0.5 are considered obstacles), based on the last k episodes.
Observations are episodic; observations from the current episode cannot be contradicted.
Meta calls expire after k observations, this is the memory length.
"""

if __name__ == "__main__":
    from mdp import make_reward_function, make_transition_function

    dim = (10,10)
    start = (4,0)
    goal = (3,8)

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
    # obstacles = [((0,3), 0.6), ((1,3), 0.7), ((2,3), 0.5), ((3,3), 1.0), ((4,3), 1.0), ((5,3), 0.65), ((6,3), 0.245), ((7,3), 0.1), ((8,3), 1.0), ((9,3), 0.0)]
    highways = [
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8),
        (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8),
        (7,8), (6,8), (5,8), (4,8), (3,5), (3,6)
    ]

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
    agent = RLMetaAgent(world, model)
    config = {
        "episodes": 200,
        "m":1,
        "lr":0.6,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps":10,
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True, obstacles=obstacles, highways=highways)
    print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")