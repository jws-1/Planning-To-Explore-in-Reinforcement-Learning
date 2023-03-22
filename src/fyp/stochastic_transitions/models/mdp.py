import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from itertools import product

class MetaAction(Enum):
    INCREASE_TRANSITION_PROBABILITY = 0
    DECREASE_TRANSITION_PROBABILITY = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


@numba.jit(nopython=True)
def value_iteration(V, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-5, max_iter=1000):

    pi = np.zeros(len(states), dtype=np.int64)

    for _ in range(max_iter):
        delta = 0

        for i in range(len(states)):
            state = states[i]
            v = V[i]

            Q = np.full(len(actions), -np.inf)
            for j in range(len(actions)):
                action = actions[j]
                for next_state in range(len(states)):
                    p = transition_function[state, action, next_state]
                    q = p * (reward_function[state, action, next_state] + discount_factor * V[next_state])
                    if q !=0.0 and Q[j] == -np.inf: Q[j] = 0.0
                    Q[j]+=q
            V[i] = np.max(Q)
            pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))
            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            break
    return V, pi


class MDP:

    def __init__(self, states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function # np array of shape (states, actions, next_state, 1[prob])
        self.reward_function = reward_function # np array of shape (states, actions, next_state, 1[reward])
        self.discount_factor = discount_factor

        if run_VI:
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
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
            self.transition_function[state, action, next_state] = N_sas[next_state] / N_sa

    def update_reward(self, state, action, next_state, reward):
        self.reward_function[state, action, next_state] = reward

    def get_rewards(self, state, action):
        return self.reward_function[state, action]

    def get_reward(self, state, action, next_state):
        return self.reward_function[state, action, next_state]

    def step(self, state, action):
        transition_probs = self.get_transition_probs(state, action)
        next_state = np.random.choice(self.states, p=transition_probs)
        return next_state, self.get_reward(state, action, next_state)

    def plan_VI(self, start, observed_sa=None, meta=None, meta_sa=None):
        self.V, self.pi = value_iteration(self.V, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=10)
        if not meta:
            return self.pi[start]

        candidate_changes_r = {(start, a, next_state, np.max(self.reward_function)) : -np.inf for (a, next_state) in product(self.actions, self.states)}
        candidate_changes_t = {(start, a, next_state) : -np.inf for (a, next_state) in product(self.actions, self.states)}

        for (s, a, s_, r) in candidate_changes_r.keys():
            if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.INCREASE_REWARD]:
                candidate_MDP = deepcopy(self)
                candidate_MDP.update_reward(s, a, s_, r)
                V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=10)
                candidate_changes_r[(s,a,s_,r)] = V_[s]

        for (s, a, s_) in candidate_changes_t.keys():
            if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.INCREASE_TRANSITION_PROBABILITY]:
                candidate_MDP = deepcopy(self)
                candidate_MDP.update_transition_prob(s, a, s_, 1.0)
                V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=10)

                candidate_changes_t[(s,a,s_)] = V_[s]
        
        best_max_r = max(candidate_changes_r.values())
        best_change_r = random.choice([c_r for c_r in candidate_changes_r.keys() if candidate_changes_r[c_r] == best_max_r])
        best_max_t = max(candidate_changes_t.values())
        best_change_t = random.choice([c_t for c_t in candidate_changes_t.keys() if candidate_changes_t[c_t] == best_max_t])
        if self.V[start] > candidate_changes_r[best_change_r] and self.V[start] > candidate_changes_t[best_change_t]:
            return self.pi[start]
        elif candidate_changes_r[best_change_r] > candidate_changes_t[best_change_t] and candidate_changes_r[best_change_r] > self.V[start]:
            _, target_action, target_state, _ = best_change_r
            return MetaAction.INCREASE_REWARD, target_action, target_state
        else:
            _, target_action, target_state = best_change_t
            return MetaAction.INCREASE_TRANSITION_PROBABILITY, target_action, target_state
