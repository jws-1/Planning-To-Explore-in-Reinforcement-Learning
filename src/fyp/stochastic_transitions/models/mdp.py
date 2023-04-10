import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from itertools import product
from collections import defaultdict

class MetaAction(Enum):
    INCREASE_TRANSITION_PROBABILITY = 0
    DECREASE_TRANSITION_PROBABILITY = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


def value_iteration(V, goal, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-10, max_iter=1000):

    pi = np.zeros(len(states), dtype=np.int64)

    for _ in range(max_iter):
        delta = 0

        for i in range(len(states)):
            state = states[i]
            if state == goal:
                V[i] = 0.0
                continue
            v = V[i]

            Q = np.full(len(actions), -np.inf)
            for j in range(len(actions)):
                action = actions[j]
                for k in range(len(states)):
                    next_state = states[k]
                    p = transition_function[state, action, next_state]
                    if p > 0.0:
                        q = p * (reward_function[state, action, next_state] + discount_factor * V[next_state])
                        if Q[j] == -np.inf: Q[j] = 0.0
                        Q[j]+=q

            V[i] = np.max(Q)
            pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))
            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            break

    return V, pi


class MDP:

    def __init__(self, states, goal, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True, reasonable_meta_transitions=None):
        self.states = states
        self.goal = goal
        self.actions = actions
        self.transition_function = transition_function # np array of shape (states, actions, next_state, 1[prob])
        self.reward_function = reward_function # np array of shape (states, actions, next_state, 1[reward])
        self.discount_factor = discount_factor
        self.reasonable_meta_transitions = reasonable_meta_transitions
        self.updated = False
        if run_VI:
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.goal, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
        else:
            self.V = np.zeros(len(self.states))
            self.pi = np.zeros(len(self.states))


    def get_transition_probs(self, state, action):
        return self.transition_function[state, action]

    def update_transition_prob(self, state, action, next_state, prob):
        if prob == 1.0:
            self.transition_function[state, action] = 0.
            self.transition_function[state, action, next_state] = 1.0
        else:
            self.transition_function[state, action, next_state] = prob
            self.transition_function[state, action] /= np.sum(self.transition_function[state, action])
    
    def update_transition_probs(self, state, action, N_sa, N_sas):
        for next_state in self.states:
            if  N_sas[next_state] / N_sa != self.transition_function[state, action, next_state]:
                self.transition_function[state, action, next_state] = N_sas[next_state] / N_sa
                self.updated = True

    def update_reward(self, state, action, next_state, reward):
        if reward != self.reward_function[state, action, next_state]:
            self.reward_function[state, action, next_state] = reward
            self.updated = True

    def get_rewards(self, state, action):
        return self.reward_function[state, action]

    def get_reward(self, state, action, next_state):
        return self.reward_function[state, action, next_state]

    def step(self, state, action):
        transition_probs = self.get_transition_probs(state, action)
        next_state = np.random.choice(self.states, p=transition_probs)
        return next_state, self.get_reward(state, action, next_state)

    def plan_VI(self, start, observed_sas=None, meta=None, meta_sas=None):
        if self.updated:
            self.V, self.pi = value_iteration(self.V, self.goal, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=10)
            self.updated = False
        if not meta:
            return self.pi[start]
        # print(self.V)
        """
        Learn the depth
            If we normally expect to get to a state through n transitions from the current state
            but it actually takes m transitions,
            then we learn that meta action
        """
        # candidate_changes_r = {state : {a : {next_state : }}}
        # candidate_changes_r = {(state, a, next_state, np.max(self.reward_function)+1) : -np.inf for (state, a, next_state) in product(self.states, self.actions, self.states)}
        # candidate_changes_t = {(state, a, next_state) : -np.inf for (state, a, next_state) in product(self.states, self.actions, self.states)}

        # candidate_changes_r = {state : (a, next_state) for (state, a, next_state) in product(self.states, self.actions, self.states)}
        # candidate_changes_t = {state : (a, next_state) for (state, a, next_state) in product(self.states, self.actions, self.states)}
        
        # candidate_changes_t = {state : {action : next_state for next_state in self.states} for action in self.actions for state in self.states}
        # candidate_changes_t = {state : (a, next_state) for  a in self.actions for next_state in self.states for state in self.states}
        # candidate_changes_r = {state : {action : next_state for next_state in self.states} for action in self.actions for state in self.states}

        changes_t = defaultdict(None)
        changes_r = defaultdict(None)
        candidate_MDP = deepcopy(self)

        state = start
        current_pi = self.pi
        current_V = self.V
        print(current_V)
        while state != self.goal:
            best_change = None
            for action in self.actions:
                for next_state in self.states:
                    if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][MetaAction.INCREASE_TRANSITION_PROBABILITY] and next_state in self.reasonable_meta_transitions[state] and candidate_MDP.transition_function[state][action][next_state] < 1.0:
                        print(f"Trying {state} {action} {next_state}, new T = 1")
                        temp_MDP = deepcopy(candidate_MDP)
                        temp_MDP.update_transition_prob(state, action, next_state, 1.0)
                        V_, pi_ = value_iteration(deepcopy(current_V), self.goal, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)
                        # print(V_)
                        # print(state, action, next_state, V_[state], current_V[state])
                        if next_state == 31:
                            print(V_[state], current_V[state])
                        if V_[state] > current_V[state]:
                            print("better")
                            best_change = (state, action, next_state, 1.0)
                            current_pi = pi_
                            current_V = V_
            if best_change is not None:
                print(f"Change! {best_change}")
                candidate_MDP.update_transition_prob(*best_change)
                changes_t[state] = best_change
            
            state, _ = candidate_MDP.step(state, current_pi[state])
            # print(state, current_pi[state])
        state = start
        
        while state != self.goal:
            best_change = None
            for action in self.actions:
                for next_state in self.states:
                    if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][MetaAction.INCREASE_REWARD] and next_state in self.reasonable_meta_transitions[state]:
                        # print(f"Trying {state} {action} {next_state}, new T = +1")
                        temp_MDP = deepcopy(candidate_MDP)
                        temp_MDP.update_reward(state, action, next_state, self.reward_function[state, action, next_state]+1)
                        V_, pi_ = value_iteration(deepcopy(current_V), self.goal, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=10)
                        if V_[state] > current_V[state]:
                            best_change = (state, action, next_state, self.reward_function[state, action, next_state]+1)
                            current_pi = pi_
                            current_V = V_
            if best_change is not None:
                candidate_MDP.update_reward(*best_change)
                changes_r[state] = best_change
            state, _ = candidate_MDP.step(state, current_pi[state])


        if changes_t[start]:
            state, action, next_state, p = changes_t[start]
            return MetaAction.INCREASE_TRANSITION_PROBABILITY, action, next_state
        elif changes_r[start]:
            state, action, next_state, r = changes_r[start]
            return MetaAction.INCREASE_REWARD, action, next_state
        else:
            return current_pi[start]            
