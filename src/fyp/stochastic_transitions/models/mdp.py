import numpy as np
import random
from copy import deepcopy
from enum import Enum
import numba
from typing import Optional
from itertools import product
import heapq
from collections import defaultdict
from ..actions import BaseMetaActions, MetaActionT, MetaActionR     
from watchpoints import watch

import heapq

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

# @numba.jit()
# def value_iteration(V, goal_states, states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-10, max_iter=1000):

#     pi = np.zeros(len(states), dtype=np.int64)
#     for _ in range(max_iter):
#         delta = 0

#         for i in range(len(states)):
#             state = states[i]
#             if state in goal_states:
#                 V[i] = 0.0
#                 continue
#             v = V[i]

#             Q = np.full(len(actions), -np.inf)
#             for j in range(len(actions)):
#                 action = actions[j]
#                 for k in range(len(states)):
#                     next_state = states[k]
#                     p = transition_function[state, action, next_state]
#                     if p > 0.0:
#                         q = p * (reward_function[state, action, next_state] + discount_factor * V[next_state])
#                         if Q[j] == -np.inf: Q[j] = 0.0
#                         Q[j]+=q

#             V[i] = np.max(Q)
#             pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))
#             delta = max(delta, abs(v - V[i]))

#         if delta < theta:
#             break
#     return V, pi



def make_deterministic(transition_function):
    # Get the number of states and actions
    num_states, num_actions, _ = transition_function.shape

    # Create an empty array to store the deterministic transition function
    deterministic_function = np.zeros((num_states, num_actions), dtype=np.int32)

    # Loop over each state and action
    for s in range(num_states):
        for a in range(num_actions):
            # Get the probabilities of transitioning to each state
            probabilities = transition_function[s, a, :]

            # Find the index of the highest probability
            max_index = np.argmax(probabilities)

            # Handle tie-breaking randomly
            indices = np.where(probabilities == probabilities[max_index])[0]
            next_state = random.choice(indices)

            # Update the deterministic transition function
            deterministic_function[s, a] = next_state

    return deterministic_function

def a_star(start, goal_states, states, actions, transition_function, reward_function, meta=False, meta_actions=None, observed_sas=None, meta_sas=None, reasonable_meta_actions=None, undiscretize_fn=None, use_learned_actions=False,action_seq_sim=None):
    transition_function = make_deterministic(transition_function)
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
            print(parent)
            return path, meta_calls
        
        # Expand the current state by applying each action
        for action in range(len(actions)):
            # Apply the action to get the next state and its probability
            best_next_state = transition_function[current, action]
            # for next_state, probability in transition_function(current, action):
            # Calculate the tentative g-score of the next state
            tentative_g_score = g_scores[current] + -reward_function[current, action, best_next_state]
            tentative_f_score = tentative_g_score +  manhattan_distance(undiscretize_fn(best_next_state), undiscretize_fn(goal_states[0]))
            # if meta:
            #     if use_learned_actions:
            #         pass
                
            #         # meta_actions_t, meta_actions_r = meta_actions
            #         # for meta_action in meta_actions_t:
            #         #     next_state = action_seq_sim(current, meta_action.action_sequence)
            #         #     if not observed_sas[current][action][next_state]:
            #         #         if not meta_sas[current][action][next_state][meta_action] and meta_action.action == action:
            #         #             f_score = g_scores[current] + -reward_function[current, action, next_state] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
            #         #             if f_score < tentative_f_score:
            #         #                 best_next_state = next_state
            #         #                 tentative_f_score = f_score
            #         #                 tentative_g_score =  g_scores[current] +  -reward_function[current, action, best_next_state]
            #         #                 meta_calls[(current, action)] = [(meta_action, action, next_state)]

            #         # for meta_action in meta_actions_r:
            #         #     for next_state in states:
            #         #         if not observed_sas[current][action][next_state] and current != next_state:
            #         #             if not meta_sas[current][action][next_state][meta_action] and meta_action.action == action:
            #         #                 f_score = g_scores[current] + -meta_action.reward + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
            #         #                 if f_score < tentative_f_score:
            #         #                     best_next_state = next_state
            #         #                     tentative_f_score = f_score
            #         #                     tentative_g_score =  g_scores[current] + -meta_action.reward
            #         #                     meta_calls[(current, action)] = [(meta_action, action, next_state)]

            #     elif reasonable_meta_actions is not None:
            #         for next_state in reasonable_meta_actions[current]:
            #             if not observed_sas[current][action][next_state]:
            #                 if not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
            #                     f_score = g_scores[current] + -np.max(reward_function) + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
            #                     if f_score < tentative_f_score:
            #                         best_next_state = next_state
            #                         tentative_f_score = f_score
            #                         tentative_g_score =  g_scores[current] + -np.max(reward_function)

            #                         if transition_function[current, action] == next_state or current == next_state:
            #                             meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
            #                         else:
            #                             meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]

            #                 elif not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
            #                     f_score = g_scores[current] + -np.max(reward_function) + manhattan_distance(undiscretize_fn(best_next_state), undiscretize_fn(goal_states[0]))
            #                     if f_score < tentative_f_score:
            #                         tentative_f_score = f_score
            #                         tentative_g_score =  g_scores[current] + -np.max(reward_function)
            #                         meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
        
            #                 elif meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
            #                     f_score = g_scores[current] + -reward_function[current, action, next_state] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
            #                     if f_score < tentative_f_score:
            #                         if transition_function[current, action] == next_state or current == next_state: continue
            #                         best_next_state = next_state
            #                         tentative_f_score = f_score
            #                         tentative_g_score =  g_scores[current] + -reward_function[current, action, next_state]
            #                         meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state)]


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





    # # Initialize the g-score and parent of each state
    # g_scores = {start: 0}
    # parent = {start: None}
    # parent_action = {start: None}
    # # Initialize the frontier with the start state and its f-score
    # frontier = [(0 + manhattan_distance(undiscretize_fn(start), undiscretize_fn(goal_states[0])), start)]
    # meta_calls = defaultdict(list)
    # # Loop until the frontier is empty or a goal state is found
    # while frontier:
    #     # Get the state with the lowest f-score from the frontier
    #     _, current = heapq.heappop(frontier)
        
    #     if current in goal_states:
    #         # print(parent, parent_action)
    #         # print(meta_calls)
    #         # Construct the optimal path by following the parent pointers
    #         path = []
    #         while current != start:
    #             path.append(parent_action[current])
    #             current = parent[current]
    #         path.reverse()
    #         print(parent)
    #         # print(path)
    #         return path, meta_calls
        
    #     # Expand the current state by applying each action
    #     for action in range(len(actions)):
    #         # Apply the action to get the next state and its probability
    #         best_next_state = None
    #         tentative_g_score = np.inf
    #         tentative_f_score = np.inf
    #         for next_state in states:
    #             p = transition_function[current, action, next_state]
    #             r = reward_function[current, action, next_state]
                

    #             # if p > 0.0:
    #             #     r = reward_function[current, action, next_state]
    #             #     g = g_scores[current] + -r
    #             #     f = g + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #             #     if f < tentative_f_score:
    #             #         tentative_g_score = g
    #             #         tentative_f_score = f
    #             #         best_next_state = next_state
            
    #         # if meta:
    #         #     if use_learned_actions:
    #         #         meta_actions_t, meta_actions_r = meta_actions
    #         #         for meta_action in meta_actions_t:
    #         #             for next_state in states:
    #         #                 if not observed_sas[current][action][next_state] and not meta_sas[current][action][next_state][meta_action]:
    #         #                     action, next_state = meta_action.action, action_seq_sim(current, meta_action.action_sequence)
    #         #                     f_score = g_scores[current] + -reward_function[current, action, next_state] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #         #                     if f_score < tentative_f_score:
    #         #                         best_next_state = next_state
    #         #                         tentative_f_score = f_score
    #         #                         tentative_g_score =  g_scores[current] +  -reward_function[current, action]
    #         #                         meta_calls[(current, action)] = [(meta_action, action, next_state)]
                    
    #         #         for meta_action in meta_actions_r:
    #         #             for next_state in states:
    #         #                 if not observed_sas[current][action][next_state] and not meta_sas[current][action][next_state][meta_action]:
    #         #                     if current != next_state:
    #         #                         f_score = g_scores[current] + (1-transition_function[current, action, next_state])*-meta_action.reward + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #         #                         if f_score < tentative_f_score:
    #         #                             best_next_state = next_state
    #         #                             tentative_f_score = f_score
    #         #                             tentative_g_score =  g_scores[current] + -meta_action.reward
    #         #                             meta_calls[(current, action)] = [(meta_action, action, next_state)] 
                
    #         #     elif reasonable_meta_actions is not None:
    #         #         for next_state in reasonable_meta_actions[current]:
    #         #             if not observed_sas[current][action][next_state]:
    #         #                 if not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
    #         #                     f_score = g_scores[current] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #         #                     if f_score < tentative_f_score:
    #         #                         best_next_state = next_state
    #         #                         tentative_f_score = f_score
    #         #                         tentative_g_score =  g_scores[current]
    #         #                         if transition_function[current, action, next_state] == 1.0 or current == next_state:
    #         #                             meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
    #         #                         else:
    #         #                             meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                            
    #         #                 elif not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
    #         #                     f_score = g_scores[current] + (1-transition_function[current, action, next_state])*-np.max(reward_function) + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #         #                     if f_score < tentative_f_score:
    #         #                         tentative_f_score = f_score
    #         #                         tentative_g_score =  g_scores[current] + (1-transition_function[current, action, next_state])*-np.max(reward_function)
    #         #                         meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
                            
    #         #                 elif meta_sas[current][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[current][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
    #         #                     f_score = g_scores[current] + -reward_function[current, action] + manhattan_distance(undiscretize_fn(next_state), undiscretize_fn(goal_states[0]))
    #         #                     if f_score < tentative_f_score:
    #         #                         if transition_function[current, action,next_state] == 1.0 or current == next_state: continue
    #         #                         best_next_state = next_state
    #         #                         tentative_f_score = f_score
    #         #                         tentative_g_score =  g_scores[current] + -reward_function[current, action]
    #         #                         meta_calls[(current, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state)]
    #         # Check if the next state is already in the g-score dictionary
    #         if best_next_state in g_scores:
    #             # If the tentative g-score is worse than the current g-score, skip this state
    #             if tentative_g_score >= g_scores[best_next_state]:
    #                 continue

    #         # Update the g-score and parent of the next state
    #         g_scores[best_next_state] = tentative_g_score
    #         parent[best_next_state] = current
    #         parent_action[best_next_state] = action
    #         # Calculate the f-score of the next state and add it to the frontier
    #         # f_score = tentative_g_score +  manhattan_distance(undiscretize_fn(best_next_state), goal_states[0])

    #         heapq.heappush(frontier, (tentative_f_score, best_next_state))
    # print(parent)
    # # If the frontier is empty and no goal state was found, return None
    # return None, float('inf')


def value_iteration(V, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-7, max_iter=1000, meta=False, meta_sas=None, observed_sas=None, reasonable_meta_actions=None, meta_actions=None, use_learned_actions=False, action_seq_sim=None):

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

            Q = np.full(len(actions), -np.inf)
            for j in range(len(actions)):
                action = actions[j]
                for k in range(len(states)):
                    next_state = states[k]
                    p = transition_function[state, action, next_state]
                    if p > 0.0:
                        q = p * (reward_function[state, action, next_state] + discount_factor * V[next_state])
                        if Q[j] == -np.inf: Q[j]=0.0
                        Q[j]+=q
           
            if meta:
                if use_learned_actions:
                    for action in range(len(actions)):
                            meta_actions_t, meta_actions_r = meta_actions                            
                            for meta_action in meta_actions_t:
                                action, next_state = meta_action.action, action_seq_sim(state, meta_action.action_sequence)
                                if not meta_sas[state][action][next_state][meta_action] and not observed_sas[state][action][next_state]:
                                
                                    q = reward_function[state, action]  + V[next_state]
                                    if q > Q[action]:
                                        Q[action] = q
                                        meta_calls[(state, action)] = [(meta_action, action, next_state)]
                            
                            for meta_action in meta_actions_r:  
                                for next_state in states:
                                    if not meta_sas[state][action][next_state][meta_action]:
                                        if state != next_state:
                                            q = meta_action.reward + V[next_state]
                                            if q > Q[action]:
                                                meta_calls[(state, action)] = [(meta_action, action, next_state)]

                elif reasonable_meta_actions is not None:
                    for action in range(len(actions)):
                        for next_state in reasonable_meta_actions[state]:
                            if not observed_sas[state][action][next_state]:
                                if not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
                                    if np.max(reward_function) == reward_function[state, action, next_state]:
                                        q = reward_function[state, action, next_state] + V[next_state]
                                        if q > Q[action]:
                                            Q[action] = q
                                            if transition_function[state, action,next_state] < 1.0 and state != next_state:
                                                meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state)]
                                    else:
                                        q = np.max(reward_function)  + V[next_state]
                                        if q > Q[action]:
                                            Q[action] = q
                                            if transition_function[state, action,next_state] < 1.0 and state != next_state:
                                                meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                                            else:
                                                meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                                        # if (transition_function[state, action, next_state] == 1.0 or state == next_state):
                                        #     meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                                        # else:
                                        #     meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                                elif not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
                                    if np.max(reward_function) != reward_function[state, action, next_state]:
                                        q = np.max(reward_function) + V[next_state]
                                        if q > Q[action]:
                                            Q[action] = q
                                            meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                                elif meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
                                    if transition_function[state, action, next_state] == 1.0 or state == next_state: continue
                                    q = reward_function[state, action, next_state] + V[next_state]
                                    if q > Q[action]:
                                        Q[action] = q
                                        meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state)]

            V[i] = max(Q)
            pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))

            delta = max(delta, abs(v - V[i]))

        if delta < theta:
            break
    
    if meta:
        return V, pi, meta_calls
    else:
        print(V, pi)
        return V, pi


# def value_iteration(V, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, theta=1e-7, max_iter=1000, meta=False, meta_sas=None, observed_sas=None, reasonable_meta_actions=None, meta_actions=None):

#     pi = np.zeros(len(states), dtype=np.int64)
#     meta_calls = defaultdict(list)
#     for _ in range(max_iter):
#         delta = 0

#         for i in range(len(states)):
#             state = states[i]

#             if state in goal_states:
#                 V[i] = 0
#                 continue

#             v = V[i]

#             Q = np.full(len(actions), 0.0)
#             for j in range(len(actions)):
#                 action = actions[j]
#                 for k in range(len(states)):
#                     next_state = states[k]
#                     p = transition_function[state, action, next_state]
#                     if p > 0.0:
#                         q = p * (reward_function[state, action, next_state] + discount_factor * V[next_state])
#                         Q[j]+=q
#             if meta:
                
#                 if reasonable_meta_actions is not None:
#                     for action in range(len(actions)):
                        
#                         # if not observed_sa[state][action]:
#                         #     r = reward_function[state, action]
#                         #     if not meta_sa[state][action][BaseMetaActions.INCREASE_REWARD] and not meta_sa[state][action][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
#                         for next_state in reasonable_meta_actions[state]:
#                             if observed_sas[state][action][next_state]: continue
#                             if not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY] and transition_function[state, action, next_state] < 1.0:
#                                 q = np.max(reward_function) + V[next_state]

#                                 if q > Q[action]:
#                                     Q[action] = q
#                                     if state == next_state:
#                                         meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
#                                     else:
#                                         meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state), (BaseMetaActions.INCREASE_REWARD, action, next_state, np.max(reward_function))]
                            
#                             elif not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
#                                 q = np.max(reward_function) + V[next_state]
#                                 if q > Q[action]:
#                                     Q[action] = q
#                                     meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_REWARD, action, np.max(reward_function))]
#                             elif meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY]:
#                                 if state != next_state:
#                                     q = np.max(reward_function) + V[next_state]
#                                     if q > Q[action]:
#                                         Q[action] = q
#                                         meta_calls[(state, action)] = [(BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state)]
#             V[i] = np.max(Q)
#             pi[i] = np.random.choice(np.array([j for j in range(len(actions)) if Q[j] == V[i]]))
#             delta = max(delta, abs(v - V[i]))

#         if delta < theta:
#             break
    
#     if meta:
#         return V, pi, meta_calls
#     else:
#         return V, pi


class MDP:

    def __init__(self, states, goal_states, actions, transition_function, reward_function, discount_factor=1.0, run_VI=True, reasonable_meta_transitions=None, planner="VI", undiscretize_fn=None):
        self.states = states
        self.goal_states = goal_states
        self.actions = actions
        self.transition_function = transition_function # np array of shape (states, actions, next_state, 1[prob])
        self.reward_function = reward_function # np array of shape (states, actions, next_state, 1[reward])
        self.discount_factor = discount_factor
        self.reasonable_meta_transitions = reasonable_meta_transitions
        self.updated = False
        self.undiscretize_fn = undiscretize_fn
        self.planner = planner
        if run_VI and planner == "VI":
            self.V, self.pi = value_iteration(np.zeros(len(self.states)), self.states, self.goal_states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
        else:
            self.V = np.zeros(len(self.states))
            self.pi = np.zeros(len(self.states))

    def get_transition_probs(self, state, action):
        return self.transition_function[state, action]

    def action_sequence(self, start_state, goal_state):
        # Create a priority queue for uniform cost search
        pq = [(0, start_state, [])]
        visited = set()

        immediate_actions = np.where(self.transition_function[start_state, :, goal_state] > 0)
        if len(immediate_actions[0] > 0):
            return [immediate_actions[0][0]]

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
        # print(start_state, goal_state, self.transition_function)
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

    def plan_astar(self, start, observed_sas=None, meta=None, meta_sas=None, meta_actions=None, use_learned_actions=False):
        # return
        plan, meta_calls = a_star(start, self.goal_states, self.states, self.actions, self.transition_function, self.reward_function, undiscretize_fn=self.undiscretize_fn, observed_sas=observed_sas, meta=meta, meta_sas=meta_sas, reasonable_meta_actions=self.reasonable_meta_transitions, action_seq_sim=self.simulate_action_sequence, meta_actions=meta_actions, use_learned_actions=use_learned_actions)
        if not meta:
            return plan[0]
        # print(a_star(start, self.goal_states, self.actions, self.transition_function, self.reward_function, undiscretize_fn=self.undiscretize_fn))
        else:
            print(start, plan)
            meta_actions = meta_calls[(start,plan[0])]
            if not meta_actions:
                return plan[0]
            else:
                return meta_actions[0]

    def plan(self, start, observed_sas=None, meta=None, meta_sas=None, meta_actions=None, use_learned_actions=False):
        if self.planner == "VI":
            return self.plan_VI(start, observed_sas=observed_sas, meta=meta, meta_sas=meta_sas, meta_actions=meta_actions, use_learned_actions=use_learned_actions)
        elif self.planner == "A*":
            return self.plan_astar(start, observed_sas=observed_sas, meta=meta, meta_sas=meta_sas, meta_actions=meta_actions, use_learned_actions=use_learned_actions)

    def plan_VI(self, start, observed_sas=None, meta=None, meta_sas=None, meta_actions=None, use_learned_actions=False):

        if self.updated and not meta:
            self.V, self.pi = value_iteration(self.V, self.states, self.goal_states, self.actions, self.transition_function, self.reward_function, self.discount_factor)
            self.updated = False
        elif meta:
            self.V, self.pi, meta_calls = value_iteration(self.V, self.states, self.goal_states, self.actions, self.transition_function, self.reward_function, self.discount_factor, meta=meta, meta_sas=meta_sas, observed_sas=observed_sas, reasonable_meta_actions=self.reasonable_meta_transitions, use_learned_actions=use_learned_actions, meta_actions=meta_actions, action_seq_sim=self.simulate_action_sequence)
            self.updated = False
            meta_actions = meta_calls[(start, self.pi[start])]
            if not meta_actions:
                return self.pi[start]
            else:
                return meta_actions[0]
        return self.pi[start]

        # if self.updated:
        #     self.V, self.pi = value_iteration(self.V, self.goal_states, self.states,  self.actions, self.transition_function, self.reward_function, self.discount_factor, max_iter=10)
        #     self.updated = False
        # if not meta or (not self.reasonable_meta_transitions and (len(meta_actions[0]) == 0 and len(meta_actions[1]) == 0)):
        #     return self.pi[start]
        
        # if self.reasonable_meta_transitions is not None:
        #     changes_t = defaultdict(None)
        #     changes_r = defaultdict(None)
        #     candidate_MDP = deepcopy(self)

        #     state = start
        #     current_pi = self.pi
        #     current_V = self.V
        #     while state not in self.goal_states:
        #         best_change = None
        #         for action in self.actions:
        #             for next_state in np.argsort(self.transition_function[state][action])[::-1]:
        #                 if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_TRANSITION_PROBABILITY] and next_state in self.reasonable_meta_transitions[state] and candidate_MDP.transition_function[state][action][next_state] < 1.0:
        #                     temp_MDP = deepcopy(candidate_MDP)
        #                     temp_MDP.update_transition_prob(state, action, next_state, 1.0)
        #                     V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

        #                     if V_[state] > current_V[state]:
        #                         best_change = (state, action, next_state, 1.0)
        #                         current_pi = pi_
        #                         current_V = V_

        #         if best_change is not None:
        #             candidate_MDP.update_transition_prob(*best_change)
        #             changes_t[state] = best_change
                
        #         state, _ = candidate_MDP.step(state, current_pi[state])
        #         print(f"T {state}")
        #     state = start
        #     while state not in self.goal_states:
        #         best_change = None
        #         for action in self.actions:
        #             for next_state in np.argsort(self.transition_function[state][action])[::-1]:
        #                 if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state][BaseMetaActions.INCREASE_REWARD] and next_state in self.reasonable_meta_transitions[state]:
        #                     temp_MDP = deepcopy(candidate_MDP)
        #                     temp_MDP.update_reward(state, action, next_state, max(temp_MDP.reward_function[state, action, next_state], np.max(candidate_MDP.reward_function)-1))
        #                     V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=10)
        #                     if V_[state] > current_V[state]:
        #                         best_change = (state, action, next_state, max(candidate_MDP.reward_function[state, action, next_state], np.max(candidate_MDP.reward_function)-1))
        #                         current_pi = pi_
        #                         current_V = V_
        #         if best_change is not None:
        #             if not changes_r.get(state):
        #                 candidate_MDP.update_reward(*best_change)
        #                 changes_r[state] = best_change
        #         state, _ = candidate_MDP.step(state, current_pi[state])
        #         print(f"R {state}")

        #     if changes_t.get(start, None):
        #         state, action, next_state, p = changes_t[start]
        #         return BaseMetaActions.INCREASE_TRANSITION_PROBABILITY, action, next_state
        #     elif changes_r.get(start, None):
        #         state, action, next_state, r = changes_r[start]
        #         return BaseMetaActions.INCREASE_REWARD, action, next_state
        #     else:
        #         return current_pi[start]         
        # else:
        #     changes_t = defaultdict(None)
        #     changes_r = defaultdict(None)
        #     candidate_MDP = deepcopy(self)

        #     meta_actions_t, meta_actions_r = meta_actions

        #     state = start
        #     current_pi = self.pi
        #     current_V = self.V

        #     while state not in self.goal_states:
        #         best_change = None
        #         for meta_action in meta_actions_t:
        #             print(meta_action)
        #             action, next_state = meta_action.action, self.simulate_action_sequence(state, meta_action.action_sequence)
        #             if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state].get(meta_action, False):
        #                 temp_MDP = deepcopy(candidate_MDP)
        #                 temp_MDP.update_transition_prob(state, action, next_state, 1.0)
        #                 V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

        #                 if V_[state] > current_V[state]:
        #                     best_change = meta_action, action, next_state
        #                     current_pi = pi_
        #                     current_V = V_
        #         if best_change is not None:
        #             meta_action, action, next_state = best_change
        #             candidate_MDP.update_transition_prob(state, action, next_state, 1.0)
        #             changes_t[state] = best_change
        #             print(f"Meta Action {meta_action} useful for {state, action, next_state}")
        #         state, _ = candidate_MDP.step(state, current_pi[state])
        #         # print(f"T {state}")

        #     state = start
        #     while state not in self.goal_states:
        #         best_change = None
        #         for meta_action in meta_actions_r:
        #             for action in self.actions:
        #                 for next_state in self.states:
        #                     if not observed_sas[state][action][next_state] and not meta_sas[state][action][next_state].get(meta_action, False):
        #                         temp_MDP = deepcopy(candidate_MDP)
        #                         temp_MDP.update_reward(state, action, next_state, meta_action.reward)
        #                         V_, pi_ = value_iteration(deepcopy(current_V), self.goal_states, temp_MDP.states, temp_MDP.actions, temp_MDP.transition_function, temp_MDP.reward_function, temp_MDP.discount_factor, max_iter=100)

        #                         if V_[state] > current_V[state]:
        #                             best_change = meta_action, action, next_state
        #                             current_pi = pi_
        #                             current_V = V_

        #         if best_change is not None:
        #             meta_action, action, next_state = best_change
        #             candidate_MDP.update_reward(state, action, next_state, meta_action.reward)
        #             changes_r[state] = best_change
        #         state, _ = candidate_MDP.step(state, current_pi[state])
        #         # print(f"R {state}")

        #     if changes_t.get(start, None):
        #         # meta_action, target_action, next_state = changes_t[start]
        #         return changes_t[start]
        #     elif changes_r.get(start, None):
        #         # meta_action, target_action, next_state = changes_r[start]
        #         return changes_r[start]
        #     else:
        #         return current_pi[start]