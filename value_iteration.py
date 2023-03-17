from mdp import MDP
from copy import deepcopy
import random
import numpy as np
from collections import defaultdict
from actions import MetaAction
"""
Rather than choosing "random" changes, need to choose the changes that would actually most benefit the planner.
"""


def evaluate_policy(mdp, V, pi, goal, max_iter=100, theta=0.00001):
    
    for _ in range(max_iter):
        delta = 0
        for state in mdp.states:
            if state == goal:
                continue
            v = V[state]
            new_v = 0
            for action in mdp.get_legal_actions(state):
                for next_state, prob in mdp.transition_function[state][pi[action]]:
                    reward = mdp.reward_function[next_state]
                    new_v += prob * (reward + mdp.discount_factor * V[next_state])
            V[state] = new_v
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V



def value_iteration(mdp, goal, max_iter=100, theta=0.00001, V = None):
    temporal_mdp = MDP(mdp.get_states(), mdp.get_actions(), deepcopy(mdp.transition_function), deepcopy(mdp.reward_function), mdp.get_discount_factor())
    temporal_mdp.prune()

    if V is None:
        V = {state: 0.0 for state in temporal_mdp.get_states()}

    pi = {state : None for state in temporal_mdp.get_states()}

    for _ in range(max_iter):
        delta = 0
        for state in temporal_mdp.states:

            if state == goal:
                continue

            v = V[state]
            
            # Compute Q-values for each action
            Q = {a: -np.inf for a in temporal_mdp.get_legal_actions(state)}
            for a in temporal_mdp.get_legal_actions(state):
                for (p, sp) in temporal_mdp.transition_function[state][a]:
                    if Q[a] == -np.inf: Q[a] = 0
                    # if p > 0.0 and state != sp:
                    #     if Q[a] == -np.inf:
                    #         Q[a] = 0.0
                    Q[a]+= p * (temporal_mdp.reward_function[sp] + temporal_mdp.discount_factor * V[sp])

            V[state] = max(Q.values())
            pi[state] = random.choice([a for a in temporal_mdp.get_legal_actions(state) if Q[a] == V[state]])
            delta = max(delta, abs(v - V[state]))
        
        # Stop if convergence threshold is met
        if delta < theta:
            break
    
    return V, pi

def plan_VI(mdp, start, goal, max_iter=1000, theta=0.0001, V=None, meta=False, o_s=None, meta_s=None, meta_sas=None):
    V, pi = value_iteration(mdp, goal, max_iter=max_iter, theta=theta)
    if not meta:
        return pi[start]

    """
    1.
        Iterate all possible changes in terms of transition probabilities,
    """
    changes_s = defaultdict(list)
    changes_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    temporal_mdp = MDP(mdp.get_states(), mdp.get_actions(), deepcopy(mdp.transition_function), deepcopy(mdp.reward_function), mdp.get_discount_factor())


    # state = start
    for state in temporal_mdp.states:

        for action in temporal_mdp.transition_function[state]:

            best_v = V[state]
            best_change = None
            best_V = V
            best_pi = pi
            for (p, sp) in temporal_mdp.transition_function[state][action]:
                                
                if p != 1.0 and sp != state and not MetaAction.INCREASE_TRANSITION_PROBABILITY in meta_sas[state][action][sp]:
                    new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), deepcopy(temporal_mdp.transition_function), deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
                    new_mdp.update_transition_prob(state, action, sp, 1.0)
                    # print(new_mdp.transition_function)
                    v_, pi_ = value_iteration(new_mdp, goal, V=deepcopy(V), max_iter=10, theta=theta)
                    # print(v_)
                    # print(state, action, V[state], v_[state], p, sp)
                    if v_[state] >  best_v:
                        best_v = v_[state]
                        best_change = (state, action, sp)
                        best_V = v_
                        best_pi = pi_

            if not sp in o_s and not MetaAction.INCREASE_REWARD in meta_s[sp]:
                new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), deepcopy(temporal_mdp.transition_function), deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
                new_mdp.update_reward(sp, abs(max(new_mdp.reward_function.values()))*2)
                v_, pi_ = value_iteration(new_mdp, goal, V=deepcopy(V),max_iter=10, theta=theta)
                if v_[state] > V[state]:
                    changes_s[state].append(MetaAction.INCREASE_REWARD)

            if best_change is not None:
                    # temporal_mdp.update_transition_prob(*best_change, 1.0)
                    # V = best_V
                    # pi = best_pi
                            # V = v_
                            # pi = pi_
                            # temporal_mdp.update_transition_prob(state, action, sp, 1.0)
                changes_sas[best_change[0]][best_change[1]][best_change[2]].append(MetaAction.INCREASE_TRANSITION_PROBABILITY)



    print(changes_s)
    print(changes_sas)
    action = pi[start]
    next_state, _ = temporal_mdp.step(start, action)
    if changes_sas[start][action][next_state]:
        return random.choice(changes_sas[start][action][next_state]), (start, action, next_state)
    if changes_s[next_state]:
        return random.choice(changes_s[next_state]), next_state, action
    
    # if best_change_tf:
    #     return best_change_tf
    # if best_change_vf:
    #     return best_change_vf, pi[start]
    return pi[start]



# def value_iteration(mdp, start, goal, max_iter=100, theta=0.00001, V=None):
#     pass


    # meta_sas_ = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # meta_s_ = defaultdict(list)



    # for action in temporal_mdp.transition_function[start]:
    #     if action == pi[start]:
    #         continue
    #     best_change = ()
    #     for (p,sp) in temporal_mdp.transition_function[start][action]:
    #         new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), deepcopy(temporal_mdp.transition_function), deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
    #         new_mdp.update_transition_prob(start, action, sp, 1.0)
    #         new_pi, new_V = value_iteration(new_mdp, start, goal, V=V)
    #         if new_pi[start] != pi[start]:
                # new_max = max(new_V[start])
                # meta_sas_[state][action][sp].append()


    # for action in temporal_mdp.transition_function[start]:
    #     if action == pi[start]:
    #         continue
    #     for (p,sp) in temporal_mdp.transition_function[start][action]:
    #         new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), deepcopy(temporal_mdp.transition_function), deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
    #         new_mdp.update_reward(sp, new_mdp.get_reward(sp)+1)
    #         new_pi, new_V = value_iteration(new_mdp, start, goal, V=V)




    # suggested_changes = []
    # for action in temporal_mdp.transition_function[start]:
    #     if action == pi[start]:
    #         continue
    #     for (p, sp) in temporal_mdp.transition_function[start][action]:
    #         new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), deepcopy(temporal_mdp.transition_function), deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
    #         new_mdp.update_reward(sp, new_mdp.get_reward(sp)+1)
    #         new_V, new_policy = 

            
    # for state in temporal_mdp.states:
    #     if state == goal:
    #         continue
    #     for action in temporal_mdp.get_legal_actions(state):
    #         if pi[state] == action: continue
    #         # Try increasing the reward for the state-action pair
    #         new_reward = temporal_mdp.reward_function[state] + 1
    #         new_mdp.reward_function[state] = new_reward
            # new_policy = value_iteration(new_mdp, start, goal, max_iter=max_iter, theta=theta)[1]
    #         if new_policy[state] != pi[state]:
    #             suggested_changes.append((state, action, "reward", new_reward))

    #         # Try changing the transition probabilities for the state-action pair
    #         for i, (p, sp) in enumerate(temporal_mdp.transition_function[state][action]):
    #             if p == 0:
    #                 continue
    #             new_p = 1 - p
    #             new_sp = random.choice(list(set(temporal_mdp.states) - {state, sp}))
    #             new_transition_function = deepcopy(temporal_mdp.transition_function)
    #             new_transition_function[state][action][i] = (new_p, new_sp)
    #             new_mdp = MDP(temporal_mdp.get_states(), temporal_mdp.get_actions(), new_transition_function, deepcopy(temporal_mdp.reward_function), temporal_mdp.get_discount_factor())
    #             new_policy = value_iteration(new_mdp, start, goal, max_iter=max_iter, theta=theta)[1]
    #             if new_policy[state] != pi[state]:
    #                 suggested_changes.append((state, action, "transition", (i, new_p, new_sp)))
        
    # print(suggested_changes)

    
    return V, pi