import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from collections import defaultdict
from itertools import permutations, combinations, product
from ..actions.meta_actions import BaseMetaActions

import heapq

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])



def a_star(start, goal_states, states, actions, transition_function, reward_function, meta=False, meta_actions=None, observed_sa=None, meta_sa=None, reasonable_meta_actions=None, undiscretize_fn=None, use_learned_actions=False,action_seq_sim=None):
    # Initialize the g-score and parent of each state
    g_scores = {start: 0}
    parent = {start: None}
    parent_action = {start: None}
   
    # Initialize the frontier with the start state and its f-score
    frontier = [(0 + manhattan_distance(undiscretize_fn(start), undiscretize_fn(goal_states[0])), start)]
    meta_calls = defaultdict(list)
    
    # Loop until the frontier is empty or a goal state is found
    while frontier:


        _, current = heapq.heappop(frontier)
        
        if current in goal_states:
            path = []
            while current != start:
                path.append(parent_action[current])
                current = parent[current]
            path.reverse()
            return path, meta_calls
        
        # Expand the current state by applying each action
        for action in range(len(actions)):
            # Apply the action to get the next state and its probability
            best_next_state = transition_function[current, action]

            # Calculate the tentative g-score of the next state
            tentative_g_score = g_scores[current] + -reward_function[current, action]
            tentative_f_score = tentative_g_score +  manhattan_distance(undiscretize_fn(best_next_state), undiscretize_fn(goal_states[0]))
            if meta:
                if use_learned_actions:
                    if not observed_sa[current][action]:
                        meta_actions_t, meta_actions_r = meta_actions
                        
                        for meta_action in meta_actions_t:
                            if not meta_sa[current][action][meta_action]:
                                action, next_state = meta_action.action, action_seq_sim(current, meta_action.action_sequence)
                                f_score = g_scores[current] + -reward_function[current, action] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
                                if f_score < tentative_f_score:
                                    best_next_state = next_state
                                    tentative_f_score = f_score
                                    tentative_g_score =  g_scores[current] +  -reward_function[current, action]
                                    meta_calls[(current, action)] = [(meta_action, action, next_state)]
                        
                        for meta_action in meta_actions_r:  
                            if not meta_sa[current][action][meta_action]:
                                for next_state in states:
                                    if current != next_state:
                                        f_score = g_scores[current] + -meta_action.reward + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
                                        if f_score < tentative_f_score:
                                            best_next_state = next_state
                                            tentative_f_score = f_score
                                            tentative_g_score =  g_scores[current] + -meta_action.reward
                                            meta_calls[(current, action)] = [(meta_action, action, next_state)]

                elif reasonable_meta_actions is not None:
                    if not observed_sa[current][action]:
                        if not meta_sa[current][action][BaseMetaActions.INCREASE_REWARD] and not meta_sa[current][action][BaseMetaActions.ADD_TRANSITION]:
                            for next_state in reasonable_meta_actions[current]:
                                f_score = g_scores[current] + -np.max(reward_function) + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))

                                if f_score < tentative_f_score:
                                    best_next_state = next_state
                                    tentative_f_score = f_score
                                    tentative_g_score =  g_scores[current] + -np.max(reward_function)

                                    if next_state == transition_function[current, action] or current == next_state:
                                        meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                                    else:
                                        meta_calls[(current, action)] = [(BaseMetaActions.ADD_TRANSITION, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]

                        elif not meta_sa[current][action][BaseMetaActions.INCREASE_REWARD] and meta_sa[current][action][BaseMetaActions.ADD_TRANSITION]:
                            f_score = g_scores[current] + -np.max(reward_function) + manhattan_distance(undiscretize_fn(best_next_state), undiscretize_fn(goal_states[0]))
                            if f_score < tentative_f_score:
                                tentative_f_score = f_score
                                tentative_g_score =  g_scores[current] + -np.max(reward_function)
                                meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                        elif meta_sa[current][action][BaseMetaActions.INCREASE_REWARD] and not meta_sa[current][action][BaseMetaActions.ADD_TRANSITION]:
                            for next_state in reasonable_meta_actions[current]:
                                f_score = g_scores[current] + -reward_function[current, action] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
                                if f_score < tentative_f_score:
                                    if next_state == transition_function[current, action] or current == next_state: continue
                                    best_next_state = next_state
                                    tentative_f_score = f_score
                                    tentative_g_score =  g_scores[current] + -reward_function[current, action]
                                    meta_calls[(current, action)] = [(BaseMetaActions.ADD_TRANSITION, action, next_state)]


            # Check if the next state is already in the g-score dictionary
            if best_next_state in g_scores:
                # If the tentative g-score is worse than the current g-score, skip this state
                if tentative_g_score >= g_scores[best_next_state]:
                    continue

            # Update the g-score and parent of the next state
            g_scores[best_next_state] = tentative_g_score
            parent[best_next_state] = current
            parent_action[best_next_state] = action

            heapq.heappush(frontier, (tentative_f_score, best_next_state))

    # If the frontier is empty and no goal state was found, return None
    return None, float('inf')


# @numba.jit(nopython=True)
def value_iteration(V, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-7, max_iter=1000, meta=False, meta_sa=None, observed_sa=None, reasonable_meta_actions=None, meta_actions=None, use_learned_actions=False, action_seq_sim=None):

    pi = np.zeros(len(states), dtype=np.int64)
    meta_calls = defaultdict(list)
    for _ in range(max_iter):
        delta = 0

        for i in range(len(states)):
            state = states[i]

            if state in goal_states:
                V[i] = 0
                continue

            v = V[i]

            Q = [reward_function[state, actions[j]] + V[transition_function[state, actions[j]]] for j in range(len(actions))]

            if meta:
                if use_learned_actions:
                    for action in range(len(actions)):
                        if not observed_sa[state][action]:
                            meta_actions_t, meta_actions_r = meta_actions
                            
                            for meta_action in meta_actions_t:
                                if not meta_sa[state][action][meta_action]:
                                    action, next_state = meta_action.action, action_seq_sim(state, meta_action.action_sequence)
                                    q = reward_function[state, action]  + V[next_state]
                                    if q > Q[action]:
                                        Q[action] = q
                                        meta_calls[(state, action)] = [(meta_action, action, next_state)]
                            
                            for meta_action in meta_actions_r:  
                                if not meta_sa[state][action][meta_action]:
                                    for next_state in states:
                                        if state != next_state:
                                            q = meta_action.reward + V[next_state]
                                            if q > Q[action]:
                                                meta_calls[(state, action)] = [(meta_action, action, next_state)]
                elif reasonable_meta_actions is not None:
                    for action in range(len(actions)):
                        if not observed_sa[state][action]:
                            r = reward_function[state, action]
                            if not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and not meta_sa[state][action][BaseMetaActions.ADD_TRANSITION]:
                                for next_state in reasonable_meta_actions[state]:
                                        q = np.max(reward_function) + V[next_state]
                                        if q > Q[action]:
                                            Q[action] = q
                                            if next_state == transition_function[state, action] or state == next_state:
                                                meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                                            else:
                                                meta_calls[(state, action)] = [(BaseMetaActions.ADD_TRANSITION, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                            elif not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and meta_sa[state][action][BaseMetaActions.ADD_TRANSITION]:
                                q = np.max(reward_function) + V[transition_function[state, action]]
                                if q > Q[action]:
                                    Q[action] = q
                                    meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                            elif meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and not meta_sa[state][action][BaseMetaActions.ADD_TRANSITION]:
                                for next_state in reasonable_meta_actions[state]:
                                        if next_state == transition_function[state, action] or state == next_state: continue
                                        q = r + V[next_state]
                                        if q > Q[action]:
                                            Q[action] = q
                                            meta_calls[(state, action)] = [(BaseMetaActions.ADD_TRANSITION, action, next_state)]

            V[i] = max(Q)
            pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))

            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            break
    
    if meta:
        return V, pi, meta_calls
    else:
        return V, pi

class D_MDP:

    def __init__(self, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True, planner="VI", reasonable_meta_transitions=None, undiscretize_fn=None):
        self.states = states
        self.actions = actions
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.reasonable_meta_transitions = reasonable_meta_transitions
        self.updated = False
        self.goal_states = goal_states
        self.planner = planner
        self.undiscretize_fn = undiscretize_fn
        if run_VI and planner == "VI":
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.states, self.goal_states, self.actions, self.transition_function, self.reward_function)
        else:
            self.V = np.zeros(len(self.states))
            self.pi = np.zeros(len(self.states))

    def get_transition(self, state, action):
        return self.transition_function[state, action]

    def update_transition(self, state, action, next_state):
        if next_state != self.transition_function[state, action]:
            self.transition_function[state, action] = next_state
            self.updated = True

    def update_reward(self, state, action, reward):
        if reward != self.reward_function[state, action]:
            self.reward_function[state, action] = reward
            self.updated = True

    def get_reward(self, state, action):
        return self.reward_function[state, action]

    def step(self, state, action):
        return self.get_transition(state, action), self.get_reward(state, action)

    def plan(self, start, observed_sa=None, meta=None, meta_sa=None, meta_actions=None, use_learned_actions=False):
        if self.planner == "VI":
            return self.plan_VI(start, observed_sa=observed_sa, meta=meta, meta_sa=meta_sa, meta_actions=meta_actions, use_learned_actions=use_learned_actions)
        elif self.planner == "A*":
            return self.plan_astar(start, observed_sa=observed_sa, meta=meta, meta_sa=meta_sa, meta_actions=meta_actions, use_learned_actions=use_learned_actions)

    def simulate_action_sequence(self, state, action_sequence):
        for action in action_sequence:
            next_state, _ = self.step(state, action)
            state = next_state
        return state


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
                next_state = self.transition_function[state, action]
                heapq.heappush(pq, (cost + 1, next_state, actions + [action]))

        # If we haven't found a path to the goal state, return None
        return None

    def plan_astar(self, start, observed_sa=None, meta=None, meta_sa=None, meta_actions=None, use_learned_actions=False):
        # return
        plan, meta_calls = a_star(start, self.goal_states, self.states, self.actions, self.transition_function, self.reward_function, undiscretize_fn=self.undiscretize_fn, observed_sa=observed_sa, meta=meta, meta_sa=meta_sa, reasonable_meta_actions=self.reasonable_meta_transitions, action_seq_sim=self.simulate_action_sequence, meta_actions=meta_actions, use_learned_actions=use_learned_actions)
        if not meta:
            return plan[0]
        # print(a_star(start, self.goal_states, self.actions, self.transition_function, self.reward_function, undiscretize_fn=self.undiscretize_fn))
        else:
            print(plan)
            meta_actions = meta_calls[(start,plan[0])]
            print(meta_actions)
            if not meta_actions:
                return plan[0]
            else:
                return meta_actions[0]

    def plan_VI(self, start, observed_sa=None, meta=None, meta_sa=None, meta_actions=None, use_learned_actions=False):

        if self.updated and not meta:
            self.V, self.pi = value_iteration(self.V, self.states, self.goal_states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
            self.updated = False
        elif meta:
            self.V, self.pi, meta_calls = value_iteration(self.V, self.states, self.goal_states, self.actions, self.transition_function, self.reward_function, self.discount_factor, meta=meta, meta_sa=meta_sa, observed_sa=observed_sa, reasonable_meta_actions=self.reasonable_meta_transitions, use_learned_actions=use_learned_actions, meta_actions=meta_actions, action_seq_sim=self.simulate_action_sequence)
            self.updated = False
            meta_actions = meta_calls[(start, self.pi[start])]
            if not meta_actions:
                return self.pi[start]
            else:
                return meta_actions[0]
        return self.pi[start]