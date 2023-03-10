from rl_agent import RLAgent
from copy import deepcopy
import numpy as np
from meta_planner import MetaPlanner
from actions import Action, MetaAction, ACTION_MODIFIERS
# from gridworld import GridWorld
import operator
from types import SimpleNamespace
from sys import exit
from collections import defaultdict
import heapq
import operator
from actions import Action, ACTION_MODIFIERS, MetaAction
from watchpoints import watch
from pprint import pprint
import random
from uct import uct_search
# from uct import MonteCarloTreeSearch


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


class RLMetaAgent(RLAgent):

    def __init__(self, env, T=None, R=None):
        self.env = env
        # self.initial_model = deepcopy(model)
        # self.model = model
        if T is not None:
            self.initial_T = T
        else:
            self.initial_T = {(i,j) : {a : [] for a in actions} for i in range(dim[0]) for j in range(dim[1])}
        if R is not None:
            self.initial_R = R
        else:
            self.initial_R = np.zeros((self.env.n, self.env.m))
        self.reset()
        pprint(self.T)
        # self.Q = np.full((*self.world.dim, 4), 0., dtype=float)

    def update_T(self, O_sas, O):
        # self.T = deepcopy(self.initial_T)
        for state in self.T.keys():
            for action in self.T[state].keys():
                for i,s in enumerate(deepcopy(self.T[state][action])):
                    entry = list(deepcopy(s))
                    if O[state][action.value][s[1]] > 0:
                        entry[0] = O_sas[state][action.value][s[1]]
                    else:
                        if self.N_sa[state][action.value] > 0.0:
                            entry[0] = self.N_sas[state][action.value][s[1]] / self.N_sa[state][action.value]
                    # print(self.N_sas[state][action.value][s[1]], self.N_sas[state][action.value])
                    self.T[state][action][i] = tuple(entry)


    def reset(self):
        self.Q = np.full((self.env.n, self.env.m, len(self.env.action_space)), 0., dtype=float)
        self.R = deepcopy(self.initial_R)
        self.R[self.env.g] = 0.
        self.N_sas = np.zeros((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m))
        self.N_sa = np.zeros((self.env.n, self.env.m, len(self.env.action_space)))
        self.T = deepcopy(self.initial_T)
        # self.planner = MetaPlanner(self.model)
    
    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.n, self.env.m))
        # T = {(i,j) : {a : [0,0] for a in Action} for i in range(self.world.dim[0]) for j in range(self.world.dim[1])}
        # observed = np.full((*self.world.dim, 4), False, dtype=bool)

        for i in range(config.episodes):
            O_sas = np.full((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m), np.inf, dtype=float)
            O = np.zeros((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m))
            O_s = np.zeros((self.env.n, self.env.m))
            meta = np.zeros((self.env.n, self.env.m, len(self.env.action_space), self.env.n, self.env.m, 6))
            reward = 0
            print(f"Meta Agent, episode {i}")

            planning = i < config.planning_steps

            state = self.env.sample()
            
            done = False
            # states[i][state]+=1

            while not done:
                # pprint(self.T)
                if planning:
                    self.update_T(O_sas, O)
                    # pprint(self.T)
                    # pprint(self.R)
                    action, target = self.plan(state, O, O_s, meta)
                else:
                    action = Action(int(np.argmax(self.Q[state])))

                if isinstance(action, Action):
                    next_state, r, done = self.env.step(action)
                    reward+=r
                    print(state, next_state, action)
                elif isinstance(action, MetaAction):
                    # print(state, target, action)
                    next_state = state
                    if action == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                        t = deepcopy(self.T)
                        for tr in t[state].keys():
                            for j, tk in enumerate(t[state][tr]):
                                if target[1] == tk[1]:
                                    entry = list(deepcopy(self.T[state][tr][j]))
                                    entry[0] = 1.0
                                    self.T[state][tr][j] = tuple(entry)
                                    print(f"Hypothesised increased T[{state}][{tr}][{target}], {self.T[state][tr]}")
                                    # self.T[state][target[0]][target[1]] = 
                                    meta[state][target[0].value][target[1]][action.value] = 1
                                    break
                    elif action == MetaAction.INCREASE_REWARD:
                        t = deepcopy(self.T)
                        for tr in t[state].keys():
                            for j, tk in enumerate(t[state][tr]):
                                if target[1] == tk[1]:
                                    # entry = list(deepcopy(self.T[state][tr][i]))
                                    # entry[0] = 1.0
                                    # self.T[state][tr][i] = tuple(entry)
                                    # print(f"Hypothesised increased T[{state}][{tr}][{target}], {self.T[state][tr]}")
                                    self.R[target[1]]+=1
                                    print(f"Hypothesised increased R[{state}][{tr}][{target[1]}], {self.R[target[1]]}")
                                    meta[state][target[0].value][target[1]][action.value] = 1
                                    break

                if isinstance(action, Action):
                    if planning: print(state, action, r, self.R[next_state])
                    old_value = self.Q[state[0]][state[1]][action.value]
                    next_max = np.max(self.Q[next_state])
                    new_value = (1 - config.lr) * old_value + config.lr * (r + config.df * next_max)
                    self.Q[state[0]][state[1]][action.value] = new_value
                    
                    self.N_sas[state][action.value][next_state]+=1
                    self.N_sa[state][action.value]+=1

                    expected_state = tuple(map(operator.add, state, ACTION_MODIFIERS[action]))

                    if state != next_state:
                        states[i][state]+=1
                        O_sas[state][action.value][next_state] = 1
                        O_s[next_state] = 1
                        if r != self.R[next_state]:
                            print(f"Updated R: {next_state}, {self.R[next_state]} -> {r}")
                            self.R[next_state] = r
                    else:
                        # Take the expected state from the model!
                        O_sas[state][action.value][expected_state] = 0
                        # if 0 <= expected_state[0] < self.env.n and 0 <= expected_state[1] < self.env.m:
                        #     O_sas[state][action.value][expected_state] = 0
                    # if 0 <= expected_state[0] < self.env.n and 0 <= expected_state[1] < self.env.m:
                    #     O_s[expected_state] = 1
                    #     O[state][action.value][expected_state] = 1
                        # if r != self.R[expected_state]:
                        #     self.R[expected_state] = r
                        #     print(f"Updated R: {expected_state}, {self.R[expected_state]} -> {r}")
                    self.update_T(O_sas, O)
                state = next_state

            rewards[i]=reward

            states[i][state]+=1
        return rewards, states


    """
    I think this should change to MCTS or something like that..., it's not exhaustive! Upper confidence trees might be the way to go.
    """
    def plan(self, s, observed_sas, observed, meta):
        """
        Maybe store the "policy" somewhere, and only update it when we see changes in the model.
        Reward needs to be modified somehow.. reward is per state, but the transitions are stochastic.
        """
        # return MonteCarloTreeSearch(self.env.g, self.T, [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).search(s)
        # return uct_search(s, self.env.g, self.T, self.R, meta=meta, observed=observed, observed_sas=observed_sas)

        V = {state: 0.0 for state in self.T.keys()}
        policy = {state: None for state in T.keys()}  
        for i in range(1000):
            delta = 0
            # Update value of each state
            for state in T.keys():
                v = V[state]
                # Calculate expected value of each action
                q_values = []
                for action in self.T[state].keys():
                    q = -np.inf
                    for prob, next_state, _ in self.T[state][action]:
                        reward = self.R[next_state]
                        if q == -np.inf and prob != 0.0: q = 0.0
                        q += prob * (reward + V[next_state])
                    q_values.append(q)
                # Update value of current state
                V[state] = max(q_values)
                policy[state] = list(T[state].keys())[np.argmax(q_values)]
                delta = max(delta, abs(v - V[state]))
            # Stop when change in value is below threshold
            if delta < 0.001:
                break
        
        def get_path(policy, start, goal):
            path = [start]
            state = start
            while state != goal:
                action = policy[state]
                next_state = list(T[state][action][0])[1]  # Assumes deterministic transitions
                path.append(next_state)
                state = next_state
            return path

        print(get_path(policy, s, self.env.g))

        return policy[s], None

        # V = np.zeros((self.env.n, self.env.m)) # initialize value function to zero
        # pi = np.random.randint(len(self.env.action_space), size=(self.env.n, self.env.m)) # initialize policy to always take action 0
        # for i in range(10000):
        #     delta = 0
        #     for i in range(self.env.n):
        #         for j in range(self.env.m):
        #             v = V[(i,j)]
        #             Q = np.zeros(len(self.env.action_space)) - np.inf
        #             for a in range(len(self.env.action_space)):
        #                 if len(self.T[(i,j)][Action(a)]) > 0:
        #                     for p, s_, r in self.T[(i,j)][Action(a)]:
        #                         Q[a] = max(Q[a], p * (self.R[s_] + V[s_]))
        #                         # if Q[a] == -np.inf:
        #                         #     Q[a] = p * (self.R[s_] + V[s_])
        #                         # else:
        #                         #     Q[a] += p * (self.R[s_] + V[s_])
        #             V[(i,j)] = np.max(Q)
        #             pi[(i,j)] = np.argmax(Q)
        #             delta = max(delta, abs(v - V[(i,j)]))
        #     if delta < 0.01:
        #         break


        
        # print(pi)
        # return Action(pi[s]), None



        # def forward(s, V, observed_sas, observed, meta):
        #     A = np.full(len(self.env.action_space), 0., dtype=float)
        #     for action, transitions in self.T[s].items():
        #         # if len(transitions) == 0:
        #         #     A[action.value] = -np.inf
        #         for p, state, r in transitions:
        #             # if not observed_sas[s][action.value][state] and not meta[s][action.value][state][MetaAction.INCREASE_TRANSITION_PROBABILITY.value]:
        #             #     p = 1.0
        #             # if not observed[state] and not meta[s][action.value][state][MetaAction.INCREASE_REWARD.value]:
        #             #     reward = self.R[state]+1.
        #             # else:
        #             #     reward = self.R[state]
        #             A[action.value]+= p * (self.R[state] + V[state])
        #     return A


        # V = np.zeros((self.env.n, self.env.m))
        # for i in range(100000):
        #     delta = 0
        #     for i in range(self.env.n):
        #         for j in range(self.env.m):
        #             A = forward((i,j), V, observed_sas, observed, meta)
        #             best = np.max(A.nonzero())
        #             delta = max(delta, np.abs(best - V[(i,j)]))
        #             V[(i,j)] = best
        #     # print(delta)
        #     if delta < 0.001:
        #         break
        # policy = {}
        # # policy = np.zeros((self.env.n, self.env.m, len(self.env.action_space)))
        # for i in range(self.env.n):
        #     for j in range(self.env.m):
        #         A = forward((i,j), V, observed_sas, observed, meta)
        #         best = np.random.choice(np.flatnonzero(A == A.max()))
        #         policy[(i,j)] = Action(best)
        #         # print((i,j), Action(best))
        #         # policy[i, j, best] = 1.0
        #         # print(policy[i,j])
        # # pprint(policy)
        # return policy[s], None
        # # state = s
        # states = [s]
        # while True:
        #     if state == self.env.g:
        #         break
        #     a = policy[state]
        #     print(state, a)
        #     state = tuple(map(operator.add, state, ACTION_MODIFIERS[Action(a)]))
        #     states.append(state)


        # print(s, policy[s[0]][s[1]])
        # print(Action(np.argmax(policy[s])))#Action(np.random.choice(np.flatnonzero(policy[s] == policy[s].max()))))



        # actions = []
        # state = s
        # while True:
        #     if state == self.env.g:
        #         break
        #     # print(Action(a), state)
        #     actions.append(Action(a))
        #     state = tuple(map(operator.add, state, ACTION_MODIFIERS[Action(a)]))

        # print(actions)
        # pri

        # W = 1
        # start = s
        # children = s
        # goal = tuple(self.env.g)
        # g = defaultdict(
        #     lambda: float("inf")
        # )  # dictionary with key: (x,y) tuple and value: length of the shortest path from (0,0) to (x,y)
        # g[tuple(start)] = 0 # set distance to source 0
        # parent = {
        #     tuple(start): None
        # }  # dictionary with key: (x,y) tuple and value is a (x,y) tuple where the value point is the parent point of the key point in terms of path traversal
        # pq = [(0, -g[tuple(start)], tuple(start))]
        # trajectory = cells_processed = 0 # statistics to keep track while A* runs

        # while pq:
        #     _, _, curr = heapq.heappop(pq)  # pop from priority queue
        #     # pretty_print_explored_region(grid, list(g.keys())) # uncomment if you want to see each step of A*
        #     trajectory += 1
        #     if curr == goal:  # check if we found our goal point
        #         break

        #     # for s in successors.keys():
        #     #     successors[s]*= self.R[s]

        #     for action, transitions in  self.T[curr].items():
        #         shuffled = deepcopy(transitions)
        #         random.shuffle(shuffled)
        #         print(curr, action, transitions)
        #         for p, state, _ in shuffled:
        #             if p < 1.0:
        #                 if p == 0.0 and observed_sas[curr][action.value][state]:
        #                     print(f"Ignoring {state}")
        #                     continue
        #                 if not observed_sas[curr][action.value][state] and not meta[curr][action.value][state][MetaAction.INCREASE_TRANSITION_PROBABILITY.value]:
        #                     p = 1.0
        #             children = state

        #             if not observed[state] and not meta[curr][action.value][state][MetaAction.INCREASE_REWARD.value]:
        #                 cost = max(abs(self.R[state]+1), 0)
        #             else:
        #                 cost = abs(self.R[state])
                    
        #             if p == 0.0:
        #                 new_g = g[curr] + cost + np.inf
        #             else:
        #                 new_g = g[curr] + cost - np.log(p)
                        
        #             if (
        #                 children not in g or new_g < g[children]
        #             ):  # only care about new undiscovered children or children with lower cost
        #                 cells_processed += 1
        #                 g[children] = new_g  # update cost
        #                 parent[children] = curr  # update parent of children to current
        #                 h_value = manhattan_distance(*children, *goal) # calculate h value
        #                 f = new_g + W * h_value  # generate f(n')
        #                 print(f, h_value , children)
        #                 heapq.heappush(
        #                     pq, (f, -new_g, children)
        #                 )  # add children to priority queue
        # else:
        #     return []

        # path = [curr]
        # while curr != tuple(start):
        #     curr = parent[curr]
        #     path.append(curr)
        # path.reverse()  # reverse the path so that it is the right order
        # print(path)

        # actions = [] # list of tuples of the form: action, posterior
        # rev_modifiers = dict((v,k) for k,v in ACTION_MODIFIERS.items())
        # for curr, next in zip(path, path[1:]):
        #     diff = tuple(map(operator.sub, next, curr))
        #     if not observed[next[0]][next[1]]:
        #         if not meta[curr[0]][curr[1]][rev_modifiers[diff].value][next[0]][next[1]][MetaAction.INCREASE_TRANSITION_PROBABILITY.value]:
        #                 for p, s, _ in self.T[curr][rev_modifiers[diff]]:
        #                     if s == next and p < 1.0:
        #                         actions.append((MetaAction.INCREASE_TRANSITION_PROBABILITY, ((rev_modifiers[diff], next))))

        #             # if [v[0] for v in self.T[curr].values() if len(v) > 0 and v[0][1] == next][0][0] < 1.0:
        #                         break
        #         if not meta[curr[0]][curr[1]][rev_modifiers[diff].value][next[0]][next[1]][MetaAction.INCREASE_REWARD.value]:
        #                 actions.append((MetaAction.INCREASE_REWARD, ((rev_modifiers[diff], next))))
        #     actions.append((rev_modifiers[diff], next))
        # return actions[0]

"""
Agent develops a belief about the state space (transition probabilities < 0.5 are considered obstacles), based on the last k episodes.
Observations are episodic; observations from the current episode cannot be contradicted.
Meta calls expire after k observations, this is the memory length.
"""

if __name__ == "__main__":
    dim = (4,4)
    start = (0,3)
    goal = (3,0)


    obstacles = [(1.0, (0,0)), (1.0, (1,2)), (1.0, (3,1)), (1.0, (3,2))]

    # obstacles = [((0,3), 0.001), ((1,3), 0.7), ((2,3), 0.5), ((3,3), 1.0), ((4,3), 1.0), ((5,3), 0.65), ((6,3), 0.245), ((7,3), 0.1), ((8,3), 1.0), ((9,3), 0.9)]
    highways = [(2,3)]#[(0,3), (0,4)]
    #T = make_transition_function(dim, obstacles) # {state : action : (p, s, ,r)}
    # R = make_reward_function(dim, highways)
    R = np.full(dim, -2, dtype=float)
    print(R)
    R[goal] = 0
    actions = [Action.RIGHT, Action.LEFT, Action.UP, Action.DOWN]
    T = {(i,j) : {a : [] for a in actions} for i in range(dim[0]) for j in range(dim[1])}
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            if 0 <= i - 1 < dim[0] and 0 <= j < dim[1]:
                if (i-1, j) == goal:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), 0.0))
                elif (i-1, j) in highways:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), -1.0))
                else:
                    T[(i,j)][Action.LEFT].append((1.0, (i-1, j), -2.0))
            else:
                T[(i,j)][Action.LEFT].append((1.0, (i, j), -10.0))
            
            if 0 <= i + 1 < dim[0] and 0 <= j < dim[1]:
                if (i+1, j) == goal: 
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), 0.0))
                elif (i+1, j) in highways:
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), -1.0))
                else:
                    T[(i,j)][Action.RIGHT].append((1.0, (i+1, j), -2.0))
            else:
                T[(i,j)][Action.RIGHT].append((1.0, (i, j), -10.0))

            if 0 <= i < dim[0] and 0 <= j - 1 < dim[1]:
                if (i, j-1) == goal:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), 0.0))
                elif (i, j-1) in highways:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), -1.0))
                else:
                    T[(i,j)][Action.DOWN].append((1.0, (i, j-1), -2.0))
            else:
                T[(i,j)][Action.DOWN].append((1.0, (i, j), -10.0))

            if 0 <= i < dim[0] and 0 <= j+1 < dim[1]:
                if (i, j+1) == goal:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), 0.0))
                elif (i, j+1) in highways:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), -1.0))
                else:
                    T[(i,j)][Action.UP].append((1.0, (i, j+1), -2.0))
            else:
                T[(i,j)][Action.UP].append((1.0, (i, j), -10.0))

    from grid import GridWorld
    world = GridWorld(dim[0], dim[1], start, goal, T, R, actions, obstacles)


    # world_rewards = np.full(dim, -2, dtype=int)
    
    # world_obstacles = {
    #     (0,3) : 0.,
    #     (2,3) : 0.5,
    #     (3,3) : 1.0,
    #     (4,3) : 1.0,
    #     (5,3) : 0.001
    # }
    # grid = GridWorld(dim, start, goal, world_obstacles, world_rewards)
    agent = RLMetaAgent(world, T = T, R = np.full(dim, -2., dtype=float))
    config = {
        "episodes": 100,
        "m":1,
        "lr":0.6,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps":20 ,
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True, obstacles=obstacles, highways=highways)
    print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")