import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional

class MetaAction(Enum):
    ADD_TRANSITION = 0
    REMOVE_TRANSITION = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


@numba.jit(nopython=True)
def value_iteration(V, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-5, max_iter=1000):

    pi = np.zeros(len(states), dtype=np.int64)

    for _ in range(max_iter):
        delta = 0

        for i in range(len(states)):
            state = states[i]
            # if state == goal:
            #     continue
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
        # self.V, self.pi = self.value_iteration()
        # self.V = np.zeros(len(self.states))
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


        changes_r = {state: {action: None for action in self.actions} for state in self.states}
        changes_t = {state: {action : None for action in self.actions} for state in self.states}

        V = self.V
        pi = self.pi
        candidate_mdp = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function), run_VI=False)    
        for state in self.states:

            for action in self.actions:

                if not observed_sa[state][action]:
                
                    if not meta_sa[state][action][MetaAction.INCREASE_REWARD]:
                        candidate_mdp_ = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function), run_VI=False)    
                        candidate_mdp_.update_reward(state, action, 1.)
                        V_, pi_ = value_iteration(deepcopy(V), candidate_mdp_.states, candidate_mdp_.actions, candidate_mdp_.transition_function, candidate_mdp_.reward_function, candidate_mdp_.discount_factor, max_iter=10)
                        if V_[state] > V[state] and pi[state] != pi_[state]:
                            V[state] = V_[state]
                            pi[state] = pi_[state]
                            changes_r[state][action] = MetaAction.INCREASE_REWARD
                            candidate_mdp.update_reward(state, action, 1.)

                    for next_state in self.states:
                        if not meta_sa[state][action][MetaAction.ADD_TRANSITION]:
                            if state == next_state:
                                continue
                            candidate_mdp_ = D_MDP(self.states, self.actions, deepcopy(self.transition_function), deepcopy(self.reward_function), run_VI=False)    
                            candidate_mdp_.update_transition(state, action, next_state)
                            V_, pi_ = value_iteration(deepcopy(V), candidate_mdp_.states, candidate_mdp_.actions, candidate_mdp_.transition_function, candidate_mdp_.reward_function, candidate_mdp_.discount_factor, max_iter=10)
                            if V_[state] > V[state] and pi[state] != pi_[state]:
                                V[state] = V_[state]
                                pi[state] = pi_[state]
                                changes_t[state][action] = (MetaAction.ADD_TRANSITION, next_state)
                                candidate_mdp.update_transition(state, action, next_state)

        action = pi[start]
        # print(changes_t[start][action])
        # print(changes_r[start][action])
        # print(action)
        # print(start, action, changes_t[start][action], changes_r[start][action])
        if changes_t[start][action]:
            meta_action, next_state = changes_t[start][action]
            return meta_action, action, next_state
        if changes_r[start][action]:
            meta_action = changes_r[start][action]
            return meta_action, action
        return pi[start]
