from rl_agent import RLAgent
from copy import deepcopy
import numpy as np
from meta_planner import MetaPlanner
from actions import Action, MetaAction, ACTION_MODIFIERS
# from gridworld import GridWorld
import operator
from types import SimpleNamespace
from sys import exit
from collections import defaultdict
import heapq
import operator
from actions import Action, ACTION_MODIFIERS, MetaAction
from watchpoints import watch
from pprint import pprint 
import random
from mdp import MDP
# from uct import uct_search
# from uct import MonteCarloTreeSearch


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


class RLMetaAgent(RLAgent):

    def __init__(self, env, MDP):
        self.env = env
        self.initial_MDP = MDP
        self.reset()

    def reset(self):
        self.Q = np.full((self.env.n, self.env.m, len(self.env.action_space)), 0., dtype=float)
        self.MDP = deepcopy(self.initial_MDP)
        self.N_sas = np.zeros((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m))
        self.N_sa = np.zeros((self.env.n, self.env.m, len(self.env.action_space)))
    
    def probs_from_observations(self, meta_sas):
        probs = defaultdict(lambda: defaultdict(dict))
        for state in self.MDP.states:
            for action in self.MDP.actions:
                for next_state in self.MDP.states:
                    if (state, action, next_state) in meta_sas:
                        continue
                    if self.N_sa[state][action.value] > 0:
                        probs[state][action][next_state] = self.N_sas[state][action.value][next_state] / self.N_sa[state][action.value]
        return probs

    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.n, self.env.m))

        for i in range(config.episodes):
            # O_sas = np.full((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m), np.inf, dtype=float)
            O_sas = {}
            O_s = set()
            meta_sas = []
            meta_s = []
            print(f"Meta Agent, episode {i}")

            planning = i < config.planning_steps

            state = self.env.sample()

            done = False

            while not done:
                if planning:
                    self.MDP.update_transition_probs(self.probs_from_observations(meta_sas))
                    pprint(self.MDP.transition_function)
                    #plan = self.MDP.plan_VI(list(O_s), meta_s, meta_sas)
                    plan = self.MDP.plan_VI(state, self.env.g)
                    if isinstance(plan, tuple):
                        if len(plan) == 3:
                            action, target_state, target_action = plan
                        elif len(plan) == 2:
                            action, target_state = plan
                    else:
                        action = plan
                else:
                    action = Action(int(np.argmax(self.Q[state])))

                if isinstance(action, Action):
                    # Step
                    next_state, reward, done = self.env.step(action)
                    print(state, action, next_state, reward)

                    # Update Q Value
                    old_value = self.Q[state[0]][state[1]][action.value]
                    next_max = np.max(self.Q[next_state])
                    new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                    self.Q[state[0]][state[1]][action.value] = new_value

                    # Count transition and actions.
                    self.N_sas[state][action.value][next_state]+=1
                    self.N_sa[state][action.value]+=1


                    if state != next_state:
                        states[i][state]+=1

                    if planning:
                        # Transition out of current state.
                        if state != next_state:
                            self.MDP.update_reward(next_state, reward)
                    O_s.add(next_state)
                    state = next_state
                else:
                    # Update MDP using Meta Action.
                    if action == MetaAction.INCREASE_REWARD:
                        meta_s.append(target_state)
                        self.MDP.update_reward(target_state, self.MDP.get_reward(target_state)+1)
                    elif action == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                        meta_sas.append((state, target_action, target_state))
                        self.MDP.update_transition_prob(state, target_action, target_state, 1.0)

                rewards[i]+=reward
            states[i][state]+=1
        pprint(self.MDP.transition_function)
        print(rewards)
        return rewards, states

if __name__ == "__main__":
    dim = (4,4)
    start = (0,3)
    goal = (3,0)


    obstacles = [(1.0, (0,0)), (1.0, (1,2)), (1.0, (3,1)), (1.0, (3,2))]

    highways = [(0,2), (2,3)]#[(0,3), (0,4)]
    R = np.full(dim, -2, dtype=float)
    R[goal] = 0
    actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    T = {(i,j) : {a : [] for a in actions} for i in range(dim[0]) for j in range(dim[1])}
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            if 0 <= i - 1 < dim[0] and 0 <= j < dim[1]:
                if (i-1, j) == goal:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), 0.0))
                elif (i-1, j) in highways:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), -1.0))
                else:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), -2.0))
            else:
                T[(i,j)][Action.LEFT].append((1.0, (i, j), -10.0))
            
            if 0 <= i + 1 < dim[0] and 0 <= j < dim[1]:
                if (i+1, j) == goal: 
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), 0.0))
                elif (i+1, j) in highways:
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), -1.0))
                else:
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), -2.0))
            else:
                T[(i,j)][Action.RIGHT].append((1.0, (i, j), -10.0))

            if 0 <= i < dim[0] and 0 <= j - 1 < dim[1]:
                if (i, j-1) == goal:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), 0.0))
                elif (i, j-1) in highways:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), -1.0))
                else:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), -2.0))
            else:
                T[(i,j)][Action.DOWN].append((1.0, (i, j), -10.0))

            if 0 <= i < dim[0] and 0 <= j+1 < dim[1]:
                if (i, j+1) == goal:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), 0.0))
                elif (i, j+1) in highways:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), -1.0))
                else:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), -2.0))
            else:
                T[(i,j)][Action.UP].append((1.0, (i, j), -10.0))

    T_MDP  = {(i,j) : {a : [] for a in actions} for i in range(dim[0]) for j in range(dim[1])}

    for i in range(dim[0]):
        for j in range(dim[1]):
            if 0 <= i - 1 < dim[0] and 0 <= j < dim[1]:
                if (i-1, j) == goal:
                    T_MDP[(i,j)][Action.LEFT].append((1.0, (i-1, j)))
                elif (i-1, j) in highways:
                    T_MDP[(i,j)][Action.LEFT].append((1.0, (i-1, j)))
                else:
                    T_MDP[(i,j)][Action.LEFT].append((1.0, (i-1, j)))
            else:
                T_MDP[(i,j)][Action.LEFT].append((1.0, (i, j)))
            
            if 0 <= i + 1 < dim[0] and 0 <= j < dim[1]:
                if (i+1, j) == goal: 
                    T_MDP[(i,j)][Action.RIGHT].append((1.0, (i+1, j)))
                elif (i+1, j) in highways:
                    T_MDP[(i,j)][Action.RIGHT].append((1.0, (i+1, j)))
                else:
                    T_MDP[(i,j)][Action.RIGHT].append((1.0, (i+1, j)))
            else:
                T_MDP[(i,j)][Action.RIGHT].append((1.0, (i, j)))

            if 0 <= i < dim[0] and 0 <= j - 1 < dim[1]:
                if (i, j-1) == goal:
                    T_MDP[(i,j)][Action.DOWN].append((1.0, (i, j-1)))
                elif (i, j-1) in highways:
                    T_MDP[(i,j)][Action.DOWN].append((1.0, (i, j-1)))
                else:
                    T_MDP[(i,j)][Action.DOWN].append((1.0, (i, j-1)))
            else:
                T_MDP[(i,j)][Action.DOWN].append((1.0, (i, j)))

            if 0 <= i < dim[0] and 0 <= j+1 < dim[1]:
                if (i, j+1) == goal:
                    T_MDP[(i,j)][Action.UP].append((1.0, (i, j+1)))
                elif (i, j+1) in highways:
                    T_MDP[(i,j)][Action.UP].append((1.0, (i, j+1)))
                else:
                    T_MDP[(i,j)][Action.UP].append((1.0, (i, j+1)))
            else:
                T_MDP[(i,j)][Action.UP].append((1.0, (i, j)))


    from grid import GridWorld
    world = GridWorld(dim[0], dim[1], start, goal, T, R, actions, obstacles)

    model = MDP(list(T.keys()), actions, T_MDP, R, 1.0)

    agent = RLMetaAgent(world, model)
    config = {
        "episodes": 100,
        "m":1,
        "lr":0.6,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps":20 ,
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True, obstacles=obstacles, highways=highways)
    print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")