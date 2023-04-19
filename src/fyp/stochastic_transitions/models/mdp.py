import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from itertools import product
import heapq
from collections import defaultdict
from ..actions import BaseMetaActions
from watchpoints import watch
@numba.jit()
def value_iteration(V, goal_states, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-10, max_iter=1000):

    pi = np.zeros(len(states), dtype=np.int64)
    while True:
        delta = 0

        for i in range(len(states)):
            state = states[i]
            if state in goal_states:
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

    def __init__(self, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True, reasonable_meta_transitions=None):
        self.states = states
        self.goal_states = goal_states
        self.actions = actions
        self.transition_function = transition_function # np array of shape (states, actions, next_state, 1[prob])
        self.reward_function = reward_function # np array of shape (states, actions, next_state, 1[reward])
        self.discount_factor = discount_factor
        self.reasonable_meta_transitions = reasonable_meta_transitions
        self.updated = False
        if run_VI:
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.goal_states, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
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
                for next_state in self.states:
                    if self.get_transition_probs(state, action)[next_state] > 0.0 and state != next_state:
                        heapq.heappush(pq, (cost + 1, next_state, actions + [action]))
        # If we haven't found a path to the goal state, return None
        print(start_state, goal_state, self.transition_function)
        return None


    def simulate_action_sequence(self, state, action_sequence):
        for action in action_sequence:
            next_state, _ = self.step(state, action)
            state = next_state
        return state

    def update_transition_prob(self, state, action, next_state, prob):
        if prob == 1.0:
            self.transition_function[state, action] = 0.
            self.transition_function[state, action, next_state] = 1.0
        else:
            self.transition_function[state, action, next_state] = prob
            self.transition_function[state, action] /= np.sum(self.transition_function[state, action])
    
    def update_transition_probs(self, state, action, N_sa, N_sas):
        self.updated = False
        for next_state in self.states:
            new_p = N_sas[next_state] / N_sa
            if new_p != self.transition_function[state, action, next_state]:
                self.transition_function[state, action, next_state] = new_p
                self.updated = True

    def update_reward(self, state, action, next_state, reward):
        self.updated = False
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
            self.V, self.pi = value_iteration(self.V, self.goal_states, self.states,  self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=10)
            self.updated = False
        if not meta or (not self.reasonable_meta_transitions and len(meta_actions) == 0):
            return self.pi[start]
        
        if self.reasonable_meta_transitions is not None:
            changes_t = defaultdict(None)
            changes_r = defaultdict(None)
            candidate_MDP = deepcopy(self)

            state = start
            current_pi = self.pi
            current_V = self.V
            while state not in self.goal_states:
                best_change = None
                for action in self.actions:
                    for next_state in np.argsort(self.transition_function[state][action])[::-1]:
                        if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY] and next_state in self.reasonable_meta_transitions[state] and candidate_MDP.transition_function[state][action][next_state] < 1.0:
                            temp_MDP = deepcopy(candidate_MDP)
                            temp_MDP.update_transition_prob(state, action, next_state, 1.0)
                            V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

                            if V_[state] > current_V[state]:
                                best_change = (state, action, next_state, 1.0)
                                current_pi = pi_
                                current_V = V_

                if best_change is not None:
                    candidate_MDP.update_transition_prob(*best_change)
                    changes_t[state] = best_change
                
                state, _ = candidate_MDP.step(state, current_pi[state])
            state = start
            while state not in self.goal_states:
                best_change = None
                for action in self.actions:
                    for next_state in np.argsort(self.transition_function[state][action])[::-1]:
                        if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and next_state in self.reasonable_meta_transitions[state]:
                            temp_MDP = deepcopy(candidate_MDP)
                            temp_MDP.update_reward(state, action, next_state, max(temp_MDP.reward_function[state, action, next_state], np.max(candidate_MDP.reward_function)-1))
                            V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=10)
                            if V_[state] > current_V[state]:
                                best_change = (state, action, next_state, max(candidate_MDP.reward_function[state, action, next_state], np.max(candidate_MDP.reward_function)-1))
                                current_pi = pi_
                                current_V = V_
                if best_change is not None:
                    if not changes_r.get(state):
                        candidate_MDP.update_reward(*best_change)
                        changes_r[state] = best_change
                state, _ = candidate_MDP.step(state, current_pi[state])

            if changes_t.get(start, None):
                state, action, next_state, p = changes_t[start]
                return BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state
            elif changes_r.get(start, None):
                state, action, next_state, r = changes_r[start]
                return BaseMetaActions.INCREASE_REWARD, action, next_state
            else:
                return current_pi[start]         
        else:
            #TODO: use meta actions
            pass
            # changes_t = defaultdict(None)

            # candidate_MDP = deepcopy(self)

            # state = start
            # current_pi = self.pi
            # current_V = self.V

            # while state not in self.goal_states:
            #     best_change = None
            #     for action in meta_actions:
            #         for next_state in np.argsort(self.transition_function[state][action.action])[::-1]:




            # for meta_action in meta_actions:
            #     action, next_state = meta_action.action, self.simulate_action_sequence(start, meta_action.action_sequence)
                
                # candidate_changes_t[(start, action, meta_action, next_state)] = -np.inf
            

            # for (s, a, m_a, s_) in candidate_changes_t.keys():
            #     if not observed_sa[s][a] and not meta_sa[s][a].get(m_a, False) and self.transition_function[s, a, s_] < 1.0:
            #         candidate_MDP = deepcopy(self)
            #         candidate_MDP.update_transition_prob(s, a, s_, 1.0)
            #         V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, goal, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor)

            #         candidate_changes_t[(s, a, m_a, s_)] = V_[s]

            # best_max_t = max(candidate_changes_t.values())
            # best_change_t = random.choice([c_t for c_t in candidate_changes_t.keys() if candidate_changes_t[c_t] == best_max_t])
            # if self.V[start] > candidate_changes_t[best_change_t]:
                
            #     return self.pi[start]
            # else:
            #     return best_change_t