from actions import Action
import random

import numpy as np

class MDP:
    def __init__(self, states, actions, transition_function, reward_function, discount_factor):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor

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

    def update_transition_prob(self, state, action, next_state, prob):
        if state not in self.transition_function:
            self.transition_function[state] = {action: [(prob, next_state)]}
        elif action not in self.transition_function[state]:
            self.transition_function[state][action] = [(prob, next_state)]
        else:
            # Remove any existing transitions to the same state
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

    def plan_VI(self, s, goal, o_s=None, meta_s=None, meta_sas=None, max_iter=1000, theta=0.0001):

        V = {state: 0 for state in self.get_states()}

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
                        q += p * (r + self.get_discount_factor() * V[s_prime])
                    q_values.append(q)
                # Choose the action that leads to the maximum value
                V[state] = max(q_values)
                delta = max(delta, abs(v - V[state]))
            # If the maximum change in value function is less than theta, we have converged
            if delta < theta:
                break
        
        # Choose the action that leads to the maximum value for the start state
        q_values = []
        for action in self.get_legal_actions(s):
            q = 0
            for p, s_prime in self.get_transition_probs(s, action):
                r = self.get_reward(s_prime)
                q += p * (r + self.get_discount_factor() * V[s_prime])
            q_values.append(q)
        max_q = max(q_values)
        max_q_actions = [self.get_actions()[i] for i in np.flatnonzero(np.isclose(q_values, max_q))]
        return random.choice(max_q_actions)
 