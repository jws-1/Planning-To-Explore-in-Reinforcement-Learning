import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from itertools import product
import heapq

# class MetaAction(Enum):
#     INCREASE_TRANSITION_PROBABILITY = 0
#     DECREASE_TRANSITION_PROBABILITY = 1
#     INCREASE_REWARD = 2
#     DECREASE_REWARD = 3


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
                for k in range(len(states)):
                    next_state = states[k]
                    p = transition_function[state, action, next_state]
                    if state != next_state and p > 0.0:
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

    def __init__(self, states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True, reasonable_meta_transitions=None):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function # np array of shape (states, actions, next_state, 1[prob])
        self.reward_function = reward_function # np array of shape (states, actions, next_state, 1[reward])
        self.discount_factor = discount_factor
        self.reasonable_meta_transitions = reasonable_meta_transitions
        self.updated = False
        if run_VI:
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
        else:
            self.V = np.zeros(len(self.states))
            self.pi = np.zeros(len(self.states))

    def get_transition_probs(self, state, action):
        return self.transition_function[state, action]

    def action_sequence(self, start_state, goal_state):
        # Create a priority queue for uniform cost search
        pq = [(0, start_state, [])]
        visited = set()

        # Run uniform cost search
        while len(pq) > 0:
            (cost, state, actions) = heapq.heappop(pq)
            if state in visited:
                continue
            visited.add(state)

            # Check if we have reached the goal state
            if state == goal_state:
                return actions

            # Add neighboring states to the priority queue
            for action in self.actions:
                for next_state, prob in zip(self.states, self.get_transition_probs(state, action)):
                    if prob > 0.0:
                        heapq.heappush(pq, (cost + 1, next_state, actions + [action]))
        # If we haven't found a path to the goal state, return None
        return None


    def simulate_action_sequence(self, state, action_sequence):
        for action in action_sequence:
            state, _ = self.step(state, action)
        return state

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

    def plan_VI(self, start, observed_sas=None, meta=None, meta_sas=None, meta_actions=None):
        if self.updated:
            self.V, self.pi = value_iteration(self.V, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=10)
            self.updated = False
        if not meta:
            return self.pi[start]
        

        candidate_changes_t = {}
        for meta_action in meta_actions:
            action, next_state = meta_action.action, self.simulate_action_sequence(start, meta_action.action_sequence)
            candidate_changes_t[(start, action, meta_action, next_state)] = -np.inf
        

        # """
        # Learn the depth
        #     If we normally expect to get to a state through n transitions from the current state
        #     but it actually takes m transitions,
        #     then we learn that meta action
        # """
        # candidate_changes_r = {(start, a, next_state, np.max(self.reward_function)) : -np.inf for (a, next_state) in product(self.actions, self.states)}
        # candidate_changes_t = {(start, a, next_state) : -np.inf for (a, next_state) in product(self.actions, self.states)}

        # for (s, a, s_, r) in candidate_changes_r.keys():
        #     if not observed_sas[s][a][s_] and not meta_sas[s][a][s_][MetaAction.INCREASE_REWARD]:
        #         candidate_MDP = deepcopy(self)
        #         candidate_MDP.update_reward(s, a, s_, r)
        #         V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=10)
        #         candidate_changes_r[(s,a,s_,r)] = V_[s]

        for (s, a, m_a, s_) in candidate_changes_t.keys():
            if not observed_sas[s][a][s_] and not meta_sas[s][a][s_][m_a] and self.transition_function[s, a, s_] < 1.0:
                candidate_MDP = deepcopy(self)
                candidate_MDP.update_transition_prob(s, a, s_, 1.0)
                V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=10)

                candidate_changes_t[(s,a,s_)] = V_[s]
        
        # best_max_r = max(candidate_changes_r.values())
        # best_change_r = random.choice([c_r for c_r in candidate_changes_r.keys() if candidate_changes_r[c_r] == best_max_r])
        # best_max_t = max(candidate_changes_t.values())
        # best_change_t = random.choice([c_t for c_t in candidate_changes_t.keys() if candidate_changes_t[c_t] == best_max_t])
        # if self.V[start] > candidate_changes_r[best_change_r] and self.V[start] > candidate_changes_t[best_change_t]:
        #     return self.pi[start]
        # elif candidate_changes_r[best_change_r] > candidate_changes_t[best_change_t] and candidate_changes_r[best_change_r] > self.V[start]:
        #     _, target_action, target_state, _ = best_change_r
        #     return MetaAction.INCREASE_REWARD, target_action, target_state
        # else:
        #     _, target_action, target_state = best_change_t
        #     return MetaAction.INCREASE_TRANSITION_PROBABILITY, target_action, target_state
