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
            if prob == 1.0:
                self.transition_function[state][action] = [(0.0, s) for p, s in self.transition_function[state][action] if s != next_state]
            else:
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

        temporal_mdp = MDP(self.get_states(), self.get_actions(), deepcopy(self.transition_function), deepcopy(self.reward_function), self.get_discount_factor())

        meta_actions = {state: [] for state in self.get_states()}

        if meta:
            for state in temporal_mdp.get_states():
                actions = deepcopy(self.actions)
                random.shuffle(actions)
                for action in temporal_mdp.actions:
                    transitions = deepcopy(temporal_mdp.get_transition_probs(state, action))
                    random.shuffle(transitions)
                    for (prob, next_state) in transitions:
                        if prob < 1.0:
                            if not (state, action, next_state) in meta_sas and not (MetaAction.INCREASE_TRANSITION_PROBABILITY, (action, next_state)) in meta_actions[next_state]:
                                temporal_mdp.update_transition_prob(state, action, next_state, 1.0)
                                meta_actions[next_state].append((MetaAction.INCREASE_TRANSITION_PROBABILITY, (action, next_state)))
                                break

                if state not in meta_s and not (MetaAction.INCREASE_REWARD, state) in meta_actions[state]:
                    temporal_mdp.update_reward(state, temporal_mdp.get_reward(state)+1)
                    meta_actions[state].append((MetaAction.INCREASE_REWARD, state))



        # Perform value iteration
        for _ in range(max_iter):
            delta = 0
            for state in temporal_mdp.get_states():
                if state == goal:
                    continue
                v = V[state]
                
                # Calculate the value for each possible action in this state
                q_values = []
                for action in temporal_mdp.get_legal_actions(state):
                    q = 0
                    for p, s_prime in temporal_mdp.get_transition_probs(state, action):
                        r = temporal_mdp.get_reward(s_prime)
                        q += p * (r + temporal_mdp.get_discount_factor() * V[s_prime])
                    q_values.append(q)
                
                # Choose the action that leads to the maximum value
                V[state] = max(q_values)
                pi[state] = Action(random.choice(np.flatnonzero(q_values == max(q_values))))
                delta = max(delta, abs(v - V[state]))
            
            # If the maximum change in value function is less than theta, we have converged
            if delta < theta:
                break
        
        print(pi)

        if meta:
            action = pi[s]
            next_state = temporal_mdp.step(s, action)
            meta_actions_ = meta_actions[next_state]
            if len(meta_actions_) == 0:
                return action
            elif meta_actions_[0][0] == MetaAction.INCREASE_REWARD:
                return *meta_actions_[0], action
            else:
                return meta_actions_[0]
        else:
            return pi[s]