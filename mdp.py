from actions import Action, MetaAction, ACTION_MODIFIERS
import random

import numpy as np
from collections import defaultdict
from copy import deepcopy
import operator

class MDP:
    def __init__(self, states, actions, transition_function, reward_function, discount_factor):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.cached_action = None

    def get_transition_probs(self, state, action):
        return self.transition_function[state][action]

    def get_reward(self, state):
        return self.reward_function[state]

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_discount_factor(self):
        return self.discount_factor

    def step(self, state, action):
        if isinstance(action, Action):
            transition_probs = self.get_transition_probs(state, action)
            probs, next_states = zip(*transition_probs)
            # Choose a random next state with probabilities given by probs
            next_state = random.choices(next_states, weights=probs)[0]
            reward = self.get_reward(next_state)
            return next_state, reward
        else:
            pass

    def get_legal_actions(self, state):
        return [action for action in self.get_actions() if self.get_transition_probs(state, action)]

    def get_legal_transitions(self, state):
        
        def t(a):
            return tuple(map(operator.add, state, ACTION_MODIFIERS[a]))

        return [t(a) for a in self.get_actions()]


    def update_transition_prob(self, state, action, next_state, prob):
        if next_state in self.get_legal_transitions(state):
        # if len([(p, s) for p, s in self.transition_function[state][action] if s != next_state]) != len(self.transition_function[state][action]):

        # if next_state in self.get_legal_transitions(state):  
        #     if state not in self.transition_function:
        #         self.transition_function[state] = {action: [(prob, next_state)]}
        #     elif action not in self.transition_function[state]:
        #         self.transition_function[state][action] = [(prob, next_state)]
        #     else:
        #         # Remove any existing transitions to the same state
            self.transition_function[state][action] = [(p, s) for p, s in self.transition_function[state][action] if s != next_state]
            self.transition_function[state][action].append((prob, next_state))

    def update_transition_probs(self, probs):
        for state in self.states:
            for action in self.actions:
                if state in probs and action in probs[state]:
                    for next_state in probs[state][action]:
                        prob = probs[state][action][next_state]
                        self.update_transition_prob(state, action, next_state, prob)

    def update_reward(self, state, reward):
        self.reward_function[state] = reward

    def update_rewards(self, rewards):
        pass

    def plan_VI(self, s, goal, meta=False, o_s=None, meta_s=None, meta_sas=None, max_iter=1000, theta=0.0001):

        V = {state: 0 for state in self.get_states()}
        pi = {state : None for state in self.get_states()}



        # meta_actions = {state: set() for state in self.get_states()}

        # meta_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        temporal_mdp = MDP(self.get_states(), self.get_actions(), deepcopy(self.transition_function), deepcopy(self.reward_function), self.get_discount_factor())

        meta_actions_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        meta_actions_s = defaultdict(set)

        for state in self.get_states():
            if not state in meta_s:
                meta_actions_s[state].add(MetaAction.INCREASE_REWARD)
                temporal_mdp.update_reward(state, temporal_mdp.get_reward(state)+1)
            for action in self.get_actions(state):
                for next_state in self.get_legal_transitions(state):
                    if not (state, action, next_state) in meta_sas:
                        meta_actions_sas[state][action][next_state].add(MetaAction.INCREASE_TRANSITION_PROBABILITY)
                        temporal_mdp.update_transition_prob(state, action, next_state, 1.0) # Ensure here that the sum of the probabilities is 1!!!!
        # Perform value iteration
        for _ in range(max_iter):
            delta = 0
            for state in self.get_states():
                if state == goal:
                    continue
                v = V[state]
                # Calculate the value for each possible action in this state
                q_values = []
                for action in self.get_legal_actions(state):
                    q = 0
                    for p, s_prime in self.get_transition_probs(state, action):
                        r = self.get_reward(s_prime)
                        """
                        Also something wrong here.../*-
                        """
                        if meta:
                            if not (state, action, s_prime) in meta_sas and p < 1.0:
                                p = 1.0
                                temporal_mdp.update_transition_prob(s, action, s_prime, 1.0)
                                meta_actions_sas[state][action][s_prime].add(MetaAction.INCREASE_TRANSITION_PROBABILITY)
                            if not s_prime in meta_s and not s_prime in o_s:
                                r+=1
                                temporal_mdp.update_reward(s_prime, r)
                                meta_actions_s[s_prime].add(MetaAction.INCREASE_REWARD)
                        q += p * (r + self.get_discount_factor() * V[s_prime])
                    q_values.append(q)
                # Choose the action that leads to the maximum value
                V[state] = max(q_values)
                pi[state] = Action(random.choice(np.flatnonzero(q_values == max(q_values))))
                delta = max(delta, abs(v - V[state]))
            # If the maximum change in value function is less than theta, we have converged
            if delta < theta:
                break
        
        # Choose the action that leads to the maximum value for the start state
        # q_values = []
        # for action in self.get_legal_actions(s):
        #     q = -np.inf
        #     for p, s_prime in self.get_transition_probs(s, action):
        #         r = self.get_reward(s_prime)
        #         q += p * (r + self.get_discount_factor() * V[s_prime])
        #     q_values.append(q)
        # max_q = max(q_values)
        # max_q_actions = [self.get_actions()[i] for i in np.flatnonzero(q_values == max_q)]
        # action = random.choice(max_q_actions)
        


        # if meta:
        #     next_state, _ = temporal_mdp.step(s, action)
            
        #     meta_action_sas = list(set(meta_actions_sas[s][action][next_state]))
        #     meta_action_s = list(set(meta_actions_s[next_state]))
        #     if len(meta_action_sas) > 0:
        #         self.cached_action = action
        #         return meta_action_sas[0], next_state, action
        #     elif len(meta_action_s) > 0:
        #         self.cached_action = action
        #         return meta_action_s[0], next_state

        print(pi)
        return pi[s]

        #meta_action = list(meta_actions[s_prime])[0]
