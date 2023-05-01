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
        # Get the state with the lowest f-score from the frontier
        _, current = heapq.heappop(frontier)
        
        if current in goal_states:
            # print(parent, parent_action)
            # print(meta_calls)
            # Construct the optimal path by following the parent pointers
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
            # for next_state, probability in transition_function(current, action):
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
            # Calculate the f-score of the next state and add it to the frontier
            # f_score = tentative_g_score +  manhattan_distance(undiscretize_fn(best_next_state), goal_states[0])

            heapq.heappush(frontier, (tentative_f_score, best_next_state))

    # If the frontier is empty and no goal state was found, return None
    return None, float('inf')

# def a_star(start, goal_states, actions, transition_function, reward_function, meta=False, meta_sa=None, observed_sa=None, reasonable_meta_actions=None, meta_actions=None, undiscretize_fn=None):
    
#     meta_calls = defaultdict(list)
#     # Initialize the frontier with the start state
#     frontier = [(manhattan_distance(undiscretize_fn(start), undiscretize_fn(goal_states[0])), start)]
#     # Initialize the explored set
#     explored = set()
#     # Initialize the dictionary for storing the path
#     path = {start: (None, None)}  # each value in the dictionary is a tuple of (previous_state, previous_action)
#     # Loop until the goal state is found or the frontier is empty
#     while frontier:
#         # Pop the state with the lowest f-score from the frontier
#         _, current_state = heapq.heappop(frontier)
#         # If the current state is a goal state, return the path
#         if current_state == goal_states[0]:
#             path_cost = 0
#             result_path = []
#             while path[current_state][0] != None:
#                 result_path.append(path[current_state][1])
#                 path_cost += -reward_function[path[current_state][0], path[current_state][1]]
#                 current_state = path[current_state][0]
#             result_path.reverse()
#             return result_path, path_cost
#         # Add the current state to the explored set
#         explored.add(current_state)

#         # Expand the current state by applying each action
#         for action in actions:
#             r = reward_function[current_state[1], action]
#             if meta:
#                 if not meta_sa[current_state[1]][action][BaseMetaActions.INCREASE_REWARD] and not observed_sa[current_state[1]][action]:
#                     r = np.max(reward_function)

#                 if not meta_sa[current_state[1]][action][BaseMetaActions.ADD_TRANSITION] and not observed_sa[current_state[1]][action]:
#                     for next_state in reasonable_meta_actions[current_state[1]]:
#                         g_score = path[current_state][1] + -r if path[current_state][1] != None else -r
#                         h_score = manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
#                         f_score = g_score + h_score

#                         if next_state not in explored and next_state not in [state[1] for state in frontier]:
#                             heapq.heappush(frontier, (f_score, next_state))
#                             path[next_state] = (current_state, action)
#                         elif next_state in [state[1] for state in frontier]:
#                             index = [state[1] for state in frontier].index(next_state)
#                             if g_score < path[next_state][1]:
#                                 path[next_state] = (current_state, action)
#                                 frontier[index] = (frontier[index][0], next_state)
                        
#             next_state = transition_function[current_state, action]
#             # If the next state is not in the explored set or the frontier, add it to the frontier
            
#             if next_state not in explored and next_state not in [state[1] for state in frontier]:
#                 g_score = path[current_state][1] + -r if path[current_state][1] != None else -r
#                 h_score = manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
#                 f_score = g_score + h_score
#                 heapq.heappush(frontier, (f_score, next_state))
#                 path[next_state] = (current_state, action)

#             # If the next state is already in the frontier, update its g-score if it is lower than the previous one
#             elif next_state in [state[1] for state in frontier]:
#                 index = [state[1] for state in frontier].index(next_state)
#                 g_score = path[current_state][1] + -r if path[current_state][1] != None else -r
#                 if g_score < path[next_state][1]:
#                     path[next_state] = (current_state, action)
#                     frontier[index] = (frontier[index][0], next_state)


#     """
#     g_score usage is incorrect here.
#     """

#     # If no path is found, return None
#     return None



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
    
        # if not meta or not self.reasonable_meta_transitions:
        #     return self.pi[start]

        # candidate_MDP = deepcopy(self)
        # changes_t = defaultdict(None)
        # changes_r = defaultdict(None)
        
        # state = start
        # current_pi = self.pi
        # current_V = self.V

        # while state not in self.goal_states:
        #     best_changes = None

        #     for action in self.actions:

        #         for next_state in self.states:
        #             if not observed_sa[state][action] and state != next_state and next_state in self.reasonable_meta_transitions[state]:
        #                 temp_MDP = deepcopy(candidate_MDP)
        #                 change_t = None
        #                 change_r = None
        #                 if self.transition_function[state, action] != next_state and not meta_sa[state][action][BaseMetaActions.ADD_TRANSITION]:
        #                     temp_MDP.update_transition(state, action, next_state)
        #                     change_t = (state, action, next_state)
        #                 if not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD]:
        #                     temp_MDP.update_reward(state, action, np.max(candidate_MDP.reward_function))
        #                     change_r = (state, action, np.max(candidate_MDP.reward_function))
        #                 if change_t or change_r:
        #                     # print(temp_MDP.transition_function[state, action])
        #                     # print(temp_MDP.reward_function[state, action])
        #                     V_, pi_ = value_iteration(deepcopy(current_V), temp_MDP.states, self.goal_states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor)
        #                     # print(f"Considering {change_t, change_r}")
        #                     # print(V_[state], current_V[state])
        #                     # print(V_)
        #                     # print(current_V)
        #                     if V_[state] > current_V[state]:
        #                         best_changes = (change_t, change_r)
        #                         current_pi = pi_
        #                         current_V = V_
        #     if best_changes is not None:
        #         best_change_t, best_change_r = best_changes
        #         candidate_MDP.update_transition(*best_change_t)
        #         candidate_MDP.update_reward(*best_change_r)
        #         changes_t[state] = best_change_t
        #         changes_r[state] = best_change_r
            
        #             # if not observed_sa[state][action] and not meta_sa[state][action][BaseMetaActions.ADD_TRANSITION] and next_state in self.reasonable_meta_transitions[state] and not self.transition_function[state, action] == next_state and state != next_state:
        #             #     print(f"Trying {state} {action} {next_state}")
        #             #     print(V_)
        #             #     print(V_[state], current_V[state])
        #             #     return

        #     # if best_change is not None:
        #     #     candidate_MDP.update_transition(*best_change)
        #     #     changes_t[state] = best_change
        #     state, _ = candidate_MDP.step(state, current_pi[state])
        #     print(state)

        

        # # while state not in self.goal_states:
        #     # best_change = None

        #     # for action in self.actions:
        #     #     next_state = candidate_MDP.transition_function[state, action] 
        #     #     if state != next_state and not observed_sa[state][action] and not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and next_state in self.reasonable_meta_transitions[state]:
        #     #         temp_MDP = deepcopy(candidate_MDP)
        #     #         temp_MDP.update_reward(state, action, candidate_MDP.reward_function[state, action]+1)
        #     #         V_, pi_ = value_iteration(deepcopy(current_V),  temp_MDP.states, self.goal_states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

        #     #         if V_[state] > current_V[state]:
        #     #             best_change = (state, action, candidate_MDP.reward_function[state, action]+1)
        #     #             current_pi = pi_
        #     #             current_V = V_
        #     #     # for next_state in self.states:
        #     #     #     if state != next_state:

        #     #     #         if not observed_sa[state][action] and not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and next_state in self.reasonable_meta_transitions[state]:
        #     #     #             temp_MDP = deepcopy(candidate_MDP)
        #     #     #             temp_MDP.update_reward(state, action, candidate_MDP.reward_function[state, action]+1)
        #     #     #             V_, pi_ = value_iteration(deepcopy(current_V),  temp_MDP.states, self.goal_states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

        #     #     #             if V_[state] > current_V[state]:
        #     #     #                 best_change = (state, action, candidate_MDP.reward_function[state, action]+1)
        #     #     #                 current_pi = pi_
        #     #     #                 current_V = V_

        #     # if best_change is not None:
        #     #     candidate_MDP.update_reward(*best_change)
        #     #     changes_t[state] = best_change
        
        #     # state, _ = candidate_MDP.step(state, current_pi[state])

        # if changes_t.get(start, None):
        #     _, target_action, target_state = changes_t[start]
        #     return BaseMetaActions.ADD_TRANSITION, target_action, target_state
        #     # return changes_t[start]
        # elif changes_r.get(start, None):
        #     _, target_action, _ = changes_r[start]
        #     return BaseMetaActions.INCREASE_REWARD, target_action, None
        # else:
        #     return current_pi[start]

        # if not meta:
        #     return self.pi[start]

        # candidate_changes_r = {(start, a, self.get_reward(start, a)+1.0) : -np.inf for a in self.actions}
        # candidate_changes_t = {(start, a, next_state) : -np.inf for (a, next_state) in product(self.actions, self.states)}

        # for (s, a, r) in candidate_changes_r.keys():
        #     if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.INCREASE_REWARD]:
        #         candidate_MDP = deepcopy(self)
        #         candidate_MDP.update_reward(s, a, r)
        #         V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=100)
        #         candidate_changes_r[(s,a,r)] = V_[s]
        
        # for (s, a, s_) in candidate_changes_t.keys():
        #     if not observed_sa[s][a] and not meta_sa[s][a][MetaAction.ADD_TRANSITION]:
        #         candidate_MDP = deepcopy(self)
        #         candidate_MDP.update_transition(s, a, s_)
        #         V_, pi_ = value_iteration(deepcopy(self.V), candidate_MDP.states, candidate_MDP.actions, candidate_MDP.transition_function, candidate_MDP.reward_function, candidate_MDP.discount_factor, max_iter=100)

        #         candidate_changes_t[(s,a,s_)] = V_[s]
        
        # best_max_r = max(candidate_changes_r.values())
        # best_change_r = random.choice([c_r for c_r in candidate_changes_r.keys() if candidate_changes_r[c_r] == best_max_r])
        # best_max_t = max(candidate_changes_t.values())
        # best_change_t = random.choice([c_t for c_t in candidate_changes_t.keys() if candidate_changes_t[c_t] == best_max_t])
        
        # if self.V[start] > candidate_changes_r[best_change_r] and self.V[start] > candidate_changes_t[best_change_t]:
        #     return self.pi[start]
        # elif candidate_changes_r[best_change_r] > candidate_changes_t[best_change_t] and candidate_changes_r[best_change_r] > self.V[start]:
        #     _, target_action, _ = best_change_r
        #     return MetaAction.INCREASE_REWARD, target_action
        # else:
        #     _, target_action, target_state = best_change_t
        #     return MetaAction.ADD_TRANSITION, target_action, target_state

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
