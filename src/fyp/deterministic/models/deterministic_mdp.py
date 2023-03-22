import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from collections import defaultdict
from itertools import permutations, combinations, product

class MetaAction(Enum):
    ADD_TRANSITION = 0
    REMOVE_TRANSITION = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


@numba.jit(nopython=True)
def value_iteration(V, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-7, max_iter=1000):

    pi = np.zeros(len(states), dtype=np.int64)

    for _ in range(max_iter):
        delta = 0

        for i in range(len(states)):
            state = states[i]
            v = V[i]

            Q = np.full(len(actions), -np.inf)
            for j in range(len(actions)):
                action = actions[j]
                Q[j] = reward_function[state, action] + (discount_factor * V[transition_function[state, action]])

            V[i] = np.max(Q)
            pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))

            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            break
    return V, pi


class D_MDP:

    def __init__(self, states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor

        if run_VI:
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
        else:
            self.V = np.zeros(len(self.states))
            self.pi = np.zeros(len(self.states))


    def get_transition(self, state, action):
        return self.transition_function[state, action]

    def update_transition(self, state, action, next_state):
        self.transition_function[state, action] = next_state
    
    def update_reward(self, state, action, reward):
        self.reward_function[state, action] = reward

    def get_reward(self, state, action):
        return self.reward_function[state, action]

    def step(self, state, action):
        return self.get_transition(state, action), self.get_reward(state, action)

    def plan_VI(self, start, observed_sa=None, meta=None, meta_sa=None):

        self.V, self.pi = value_iteration(self.V, self.states, self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=100)
        if not meta:
            return self.pi[start]

        candidate_changes_r = {(start, a, self.get_reward(start, a)+1.0) : -np.inf for a in self.actions}
        candidate_changes_t = {(start, a, next_state) : -np.inf for (a, next_state) in product(self.actions, self.states)}

        for (s, a, r) in candidate_changes_r.keys():
            if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.INCREASE_REWARD]:
                candidate_MDP = deepcopy(self)
                candidate_MDP.update_reward(s, a, r)
                V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=100)
                candidate_changes_r[(s,a,r)] = V_[s]
        
        for (s, a, s_) in candidate_changes_t.keys():
            if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.ADD_TRANSITION]:
                candidate_MDP = deepcopy(self)
                candidate_MDP.update_transition(s, a, s_)
                V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=100)

                candidate_changes_t[(s,a,s_)] = V_[s]
        
        best_max_r = max(candidate_changes_r.values())
        best_change_r = random.choice([c_r for c_r in candidate_changes_r.keys() if candidate_changes_r[c_r] == best_max_r])
        best_max_t = max(candidate_changes_t.values())
        best_change_t = random.choice([c_t for c_t in candidate_changes_t.keys() if candidate_changes_t[c_t] == best_max_t])
        
        if self.V[start] > candidate_changes_r[best_change_r] and self.V[start] > candidate_changes_t[best_change_t]:
            return self.pi[start]
        elif candidate_changes_r[best_change_r] > candidate_changes_t[best_change_t] and candidate_changes_r[best_change_r] > self.V[start]:
            _, target_action, _ = best_change_r
            return MetaAction.INCREASE_REWARD, target_action
        else:
            _, target_action, target_state = best_change_t
            return MetaAction.ADD_TRANSITION, target_action, target_state

        # for s in self.states:
        #     for a in self.actions:
        #         for next_state in self.states:
        #             changes_t.append(s, a, next_state)


        # s_a_pairs = set((s,a) for s, a, _ in changes_t)
        # s_a_pairs_changes = defaultdict(list)
        
        # for s, a, next_state in changes_t:
        #     s_a_pairs_changes[(s,a)].append(next_state)
        
        # permutations = [[]]
        # for (s,a) in s_a_pairs:
        #     new_permutations = []
        #     for perm in permutations:
        #         if not any(s_ == s and a == a_ for s_, a_, _ in perm):
        #             for 


        #             changes.append(("T", s, a, next_state))
        #         changes.append(("R", s, a, self.reward_function[s][a]+1))

        # print(changes)
        # print(len(changes), len(self.states), len(self.actions))

        # changes_r = {state: {action: None for action in self.actions} for state in self.states}
        # changes_t = {state: {action : None for action in self.actions} for state in self.states}

        # V = self.V
        # pi = self.pi

        # for state in self.states:

        #     for action in self.actions:

        #         if not observed_sa[state][action]:
                
        #             if not meta_sa[state][action][MetaAction.INCREASE_REWARD]:
        #                 candidate_mdp_ = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function), run_VI=False)    
        #                 candidate_mdp_.update_reward(state, action, 1.)
        #                 V_, pi_ = value_iteration(deepcopy(V), candidate_mdp_.states, candidate_mdp_.actions, candidate_mdp_.transition_function, candidate_mdp_.reward_function, candidate_mdp_.discount_factor, max_iter=10)
        #                 if V_[state] > V[state] and pi[state] != pi_[state]:
        #                     V[state] = V_[state]
        #                     pi[state] = pi_[state]
        #                     changes_r[state][action] = MetaAction.INCREASE_REWARD

        #             for next_state in self.states:
        #                 if not meta_sa[state][action][MetaAction.ADD_TRANSITION]:
        #                     if state == next_state:
        #                         continue
        #                     candidate_mdp_ = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function), run_VI=False)    
        #                     candidate_mdp_.update_transition(state, action, next_state)
        #                     V_, pi_ = value_iteration(deepcopy(V), candidate_mdp_.states, candidate_mdp_.actions, candidate_mdp_.transition_function, candidate_mdp_.reward_function, candidate_mdp_.discount_factor, max_iter=10)
        #                     if V_[state] > V[state] and pi[state] != pi_[state]:
        #                         V[state] = V_[state]
        #                         pi[state] = pi_[state]
        #                         changes_t[state][action] = (MetaAction.ADD_TRANSITION, next_state)

        # action = pi[start]
        # if changes_t[start][action]:
        #     meta_action, next_state = changes_t[start][action]
        #     return meta_action, action, next_state
        # if changes_r[start][action]:
        #     meta_action = changes_r[start][action]
        #     return meta_action, action
        # return pi[start]
