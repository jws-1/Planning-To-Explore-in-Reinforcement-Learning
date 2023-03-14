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
        # return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
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
                self.transition_function[state][action] = [(0.0, s) for _, s in self.transition_function[state][action] if s != next_state]
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


    def prune(self):
        for state in self.states:
            for action in self.actions:
                self.transition_function[state][action] = [(p,s) for p,s in self.transition_function[state][action] if p > 0.0]

    def update_reward(self, state, reward):
        self.reward_function[state] = reward

    def update_rewards(self, rewards):
        pass

    def reachable(self, state):
        """
        Determines if a given state can be reached by any other state, excluding itself of course,
        taking into account the probability of the transition.
        """
        visited = set()
        stack = [state]

        while stack:
            curr_state = stack.pop()
            visited.add(curr_state)

            for action in self.actions:
                for (prob, next_state) in self.transition_function[curr_state][action]:
                    if next_state not in visited and prob > 0:
                        if next_state == state:
                            return True
                        else:
                            stack.append(next_state)

        return False

    def plan_VI(self, s, goal, meta=False, o_s=None, meta_s=None, meta_sas=None, max_iter=100, theta=0.0001):
        """
        Maybe remove everything that is probability 0?
        """
        temporal_mdp = MDP(self.get_states(), self.get_actions(), deepcopy(self.transition_function), deepcopy(self.reward_function), self.get_discount_factor())

        # meta_actions = {state: [] for state in self.get_states()}

        o_s = list(o_s)

        meta_sas_ = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        meta_s_ = defaultdict(list)
        
        if meta:
            for state in temporal_mdp.get_states():
                actions = deepcopy(self.actions)
                random.shuffle(actions)
                for action in actions:
                    transitions = deepcopy(temporal_mdp.get_transition_probs(state, action))
                    random.shuffle(transitions)
                    for (prob, next_state) in transitions:
                        if prob < 1.0 and state != next_state:
                            #meta_call = (MetaAction.INCREASE_TRANSITION_PROBABILITY, (state, action, next_state))
                            #if not meta_call in meta_sas and not meta_call in meta_actions[next_state]:
                            if not MetaAction.INCREASE_TRANSITION_PROBABILITY in meta_sas[state][action][next_state] and not MetaAction.INCREASE_TRANSITION_PROBABILITY in meta_sas_[state][action][next_state]: 
                                temporal_mdp.update_transition_prob(state, action, next_state, 1.0)
                                meta_sas_[state][action][next_state].append(MetaAction.INCREASE_TRANSITION_PROBABILITY)
                                break

                if MetaAction.INCREASE_REWARD not in meta_s[state] and not MetaAction.INCREASE_REWARD in meta_s_[state] and not state in o_s:
                    temporal_mdp.update_reward(state, temporal_mdp.get_reward(state)+1)
                    meta_s_[state].append(MetaAction.INCREASE_REWARD)

        temporal_mdp.prune()
        V = {state: 0.0 for state in temporal_mdp.get_states()}
        pi = {}
        for _ in range(max_iter):
            delta = 0
            for state in temporal_mdp.states:

                if state == goal:
                    continue

                v = V[state]
                
                # Compute Q-values for each action
                Q = {a: sum(p * (temporal_mdp.reward_function[sp] + temporal_mdp.discount_factor * V[sp])
                            for (p, sp) in temporal_mdp.transition_function[state][a]
                            if p > 0) for a in temporal_mdp.get_legal_actions(state)}
                
                # Update value function with maximum Q-value
                V[state] = max(Q.values()) if Q else 0
                # if s == state:
                #     print([a for a in temporal_mdp.get_legal_actions(state) if Q[a] == V[state]])
                #     import sys
                #     sys.exit(69)
                pi[state] = random.choice([a for a in temporal_mdp.get_legal_actions(state) if Q[a] == V[state]])
                # Check for convergence
                delta = max(delta, abs(v - V[state]))
            
            # Stop if convergence threshold is met
            if delta < theta:
                break

        # Compute optimal policy based on final value function
        # pi = {}
        # for state in self.states:
        #     Q = {a: sum(p * (temporal_mdp.reward_function[sp] + temporal_mdp.discount_factor * V[sp])
        #                 for (p, sp) in temporal_mdp.transition_function[state][a]
        #                 if p > 0) for a in self.get_legal_actions(state)}
        #     pi[state] = max(temporal_mdp.get_legal_actions(state), key=lambda a: Q[a])

        print(pi[s])
        if meta:
            action = pi[s]
            next_state, _ = temporal_mdp.step(s, action)
            meta_actions_sas = meta_sas_[s][action][next_state]
            if len(meta_actions_sas):
                return meta_actions_sas[0], (s, action, next_state)
            meta_actions_s = meta_s_[next_state]
            if len(meta_actions_s):
                return meta_actions_s[0], next_state, action
            else:
                return action
        else:
            return pi[s]

        # # Perform value iteration
        # for _ in range(max_iter):
        #     delta = 0

        #     # For each state, compute the new value function based on the current value function and the transition probabilities and rewards
        #     for state in temporal_mdp.get_states():

        #         if state == goal:
        #             continue
                

        #         # Get the legal actions for the current state
        #         legal_actions = temporal_mdp.get_legal_actions(state)
                
        #         # Compute the Q-value for each legal action
        #         Q = {}
        #         for action in legal_actions:
        #             q = 0.0
        #             # Compute the expected value of the next state and add it to the Q-value
        #             for prob, next_state in temporal_mdp.get_transition_probs(state, action):
        #                 q += prob * (temporal_mdp.get_reward(next_state) + temporal_mdp.discount_factor * V[next_state])
        #                 Q[action] = q
        #         # Update the value function for the current state to be the maximum Q-value
        #         if Q:
        #             max_key = max(Q, key=lambda k: (Q[k], np.random.random()))
        #             new_v = Q[max_key]
        #             # new_v = max(Q)
        #             delta = max(delta, abs(new_v - V[state]))
        #             V[state] = new_v
        #             pi[state] = max_key

        #     # for state in temporal_mdp.get_states():
        #     #     if state == goal:
        #     #         continue
        #     #     v = V[state]
                
        #     #     # Calculate the value for each possible action in this state
        #     #     q_values = []
        #     #     for action in temporal_mdp.get_legal_actions(state):
        #     #         q = -np.inf
        #     #         for p, s_prime in temporal_mdp.get_transition_probs(state, action):
        #     #             if p > 0.0 and q == -np.inf: q = 0.0
        #     #             r = temporal_mdp.get_reward(s_prime)
        #     #             q += p * (r + temporal_mdp.get_discount_factor() * V[s_prime])
        #     #         q_values.append(q)
        #     #     # Choose the action that leads to the maximum value
        #     #     V[state] = max(q_values)
        #     #     delta = max(delta, abs(v - V[state]))
            
        #     # If the maximum change in value function is less than theta, we have converged
        #     if delta < theta:
        #         break