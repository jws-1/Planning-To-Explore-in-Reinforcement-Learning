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
    
    def probs_from_observations(self):
        probs = defaultdict(lambda: defaultdict(dict))
        for state in self.MDP.states:
            for action in self.MDP.actions:
                for next_state in self.MDP.states:
                    # if (MetaAction.INCREASE_TRANSITION_PROBABILITY, (state, action, next_state)) in meta_sas:
                    #     continue
                    if self.N_sa[state][action.value] > 0:
                        if next_state in [t[1] for t in self.MDP.transition_function[state][action]]:#self.MDP.get_legal_transitions(state):
                            probs[state][action][next_state] = self.N_sas[state][action.value][next_state] / self.N_sa[state][action.value]
        return probs

    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.n, self.env.m))
        actions = [[] for i in range(config.episodes)]
        watch(self.MDP.transition_function)
        for i in range(config.episodes):
            # O_sas = np.full((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m), np.inf, dtype=float)

            print(f"Meta Agent, episode {i}")

            planning = i < config.planning_steps

            state = self.env.sample()

            O_sas = {}
            O_s = set([state])
            meta_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            meta_s = defaultdict(list)

            done = False
            next_action = None
            verify_reward = None
            while not done:

                if next_action is not None:
                    action = next_action
                    next_action = None
                    print("Using NA")
                elif planning:
                    """
                    1. Transition probabilities are not getting updated correctly.
                    2. Need to ensure that meta action is followed by a relevant action. Maybe store the current policy?
                    """
                    
                    # pprint(self.MDP.transition_function)
                    plan = self.MDP.plan_VI(state, self.env.g, True, O_s, meta_s, meta_sas)
                    if isinstance(plan, tuple):
                        if plan[0] == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                            action, (_, target_action, target_state) = plan
                            next_action = target_action
                        elif plan[0] == MetaAction.INCREASE_REWARD:
                            action, target_state, next_action = plan
                            verify_reward = target_state
                    else:
                        print(self.MDP.transition_function)
                        action = plan
                else:
                    action = Action(int(np.argmax(self.Q[state])))

                if isinstance(action, Action):
                    actions[i].append((state, action))
                    # Step
                    next_state, reward, done = self.env.step(action)
                    print(state, action, next_state)
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
                        if verify_reward is not None and next_state != verify_reward:
                            print(f"Didn't transition to {verify_reward}, so decreasing.")
                            self.MDP.update_reward(verify_reward, self.MDP.get_reward(verify_reward)-1)
                            verify_reward = None
                        self.MDP.update_transition_probs(self.probs_from_observations())
                    #O_s.add(next_state)
                    state = next_state

                else:
                    # Update MDP using Meta Action.
                    if action == MetaAction.INCREASE_REWARD:
                        actions[i].append((state, action, target_state))
                        print(action, target_state)
                        meta_s[target_state].append(action)
                        self.MDP.update_reward(target_state, self.MDP.get_reward(target_state)+1)
                    elif action == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                        actions[i].append((state, action, target_action, target_state))
                        print(action, state, target_action, target_state)
                        meta_sas[state][target_action][target_state].append(action)
                        self.MDP.update_transition_prob(state, target_action, target_state, 1.0)
                    reward = 0

                rewards[i]+=reward
            
            states[i][state]+=1

        with open("out.txt", "w") as fp:
            pprint(actions, fp)

        with open("transition.txt", "w") as fp:
            pprint(self.MDP.transition_function, fp)
            pprint(self.env.T, fp)

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
        "planning_steps":10 ,
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True, obstacles=obstacles, highways=highways)
    print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")