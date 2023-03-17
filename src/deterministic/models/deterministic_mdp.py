import numpy as np
import random
from copy import deepcopy
from enum import Enum

class MetaAction(Enum):
    ADD_TRANSITION = 0
    REMOVE_TRANSITION = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


class D_MDP:

    def __init__(self, states, actions, transition_function, reward_function, discount_factor):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.pi = {state: None for state in states}
        self.V = {state: -np.inf for state in states}

    def get_transition(self, state, action):
        return self.transition_function[state][action]

    def update_transition(self, state, action, next_state):
        self.transition_function[state][action] = next_state
    
    def update_reward(self, state, action, reward):
        self.reward_function[state][action] = reward

    def get_reward(self, state, action):
        return self.get_reward[state][action]

    def step(self, state, action):
        return self.get_transition(state, action), self.get_reward(state, action)

    def value_iteration(self, theta=1e-4, max_iter=1000, V = None):
        if V is None:
            V = {state: 0 for state in self.states}
        pi = {state: None for state in self.states}
        
        for _ in range(max_iter):
            delta = 0
            for state in self.mdp.states:
                v = V[state]

                Q = {action: -np.inf for action in self.actions}
                for action in self.actions:
                    Q[action] = self.get_reward(state, action) + (self.discount_factor * V[self.get_transition(state, action)])
                
                V[state] = max(Q.values())
                pi[state] = random.choice([action for action in self.actions if Q[action] == V[state]])

                delta = max(delta, abs(v - V[state]))

            if delta < theta:
                break
        
        return V, pi

    def plan_VI(self, start, observed_sa=None, meta=None, meta_sa=None, meta_sas=None):
        V, pi = self.value_iteration()

        if not meta:
            return pi[start]
        
        for state in self.states:
            for action in self.actions:
                if not observed_sa[state][action]:
                    if not meta_sa[state][action][meta_sa][MetaAction.INCREASE_REWARD]:
                        candidate_mdp = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function))
                        candidate_mdp.update_reward(state, action, self.get_reward(state, action)+1)
                        V_, pi_ = self.value_iteration(max_iter=10, V=V)
                        if V_[state] > V[state]:
                            self.update_reward(state, action, self.get_reward(state, action)+1)
                            meta_sa[state][action][meta_sa][MetaAction.INCREASE_REWARD] = True
                            V = V_
                            pi = pi_
                    

                    best_V = V
                    best_pi = pi
                    best_change = None
                    for next_state in self.states:
                        if not meta_sas[state][action][next_state]:
                            candidate_mdp = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function))
                            candidate_mdp.update_reward(state, action, self.get_reward(state, action)+1)
                            V_, pi_ = self.value_iteration(max_iter=10, V=V)
                            if V_[state] > best_V[state]:
                                best_V = V_
                                best_pi = pi_
                                best_change = next_state
                    if best_change:
                        self.update_transition(state, action, best_change)
                        V = best_V
                        pi = best_pi
                        meta_sas[state][action][best_change][MetaAction.ADD_TRANSITION] = True
    
        return pi[start]