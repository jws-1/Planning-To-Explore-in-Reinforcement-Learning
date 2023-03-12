from actions import Action, ACTION_MODIFIERS, MetaAction
import operator
from copy import deepcopy
import numpy as np

class GridWorld():

    def __init__(self, dim, start, goal, T, R):
        self.dim = dim
        self.start = start
        self.goal = goal
        self.T = T
        self.T_ = {}
        self.R = R
        np.random.seed(69)
    
    def sample(self):
        self.T_ = {(i,j) : {} for i in range(self.dim[0]) for j in range(self.dim[1])}
        for s in self.T.keys():
            for a in self.T[s].keys():
                s_, p = self.T[s][a]
                if np.random.rand() < p:
                    self.T_[s][a] = s_
                else:
                    self.T_[s][a] = s

    def action(self, s, a):
        s_ = self.T_[s][a]
        if s_ == s:
            return -10.0, s_
        else:
            return self.R[s], s_


    def meta_action(self, s, a, target):
        if a == MetaAction.INCREASE_REWARD:
            self.R[target] = -1.
            # self.R[target
        elif a == MetaAction.DECREASE_REWARD:
            self.R[target] = -2.
        elif a == MetaAction.INCREASE_TRANSITION_PROBABILITY:
            x,y = target
            if 0 <= x-1 < self.dim[0] and 0 <= y < self.dim[1]:
                self.T[(x-1, y)][Action.RIGHT] = ((x, y), 1.0)
            if 0 <= x+1 < self.dim[0] and 0 <= y < self.dim[1]:
                self.T[(x+1, y)][Action.LEFT] = ((x,y), 1.0)
            if 0 <= x < self.dim[0] and 0 <= y-1 < self.dim[1]:
                self.T[(x, y-1)][Action.UP] = ((x,y), 1.0)
            if 0 <= x < self.dim[0] and 0 <= y+1 < self.dim[1]:
                self.T[(x, y+1)][Action.DOWN] = ((x,y), 1.0)
        elif a == MetaAction.DECREASE_TRANSITION_PROBABILITY:
            x,y = target
            if 0 <= x-1 < self.dim[0] and 0 <= y < self.dim[1]:
                self.T[(x-1, y)][Action.RIGHT] = ((x, y), 0.0)
            if 0 <= x+1 < self.dim[0] and 0 <= y < self.dim[1]:
                self.T[(x+1, y)][Action.LEFT] = ((x,y), 0.0)
            if 0 <= x < self.dim[0] and 0 <= y-1 < self.dim[1]:
                self.T[(x, y-1)][Action.UP] = ((x,y), 0.0)
            if 0 <= x < self.dim[0] and 0 <= y+1 < self.dim[1]:
                self.T[(x, y+1)][Action.DOWN] = ((x,y), 0.0)
        return 0, s


    def feasible_state(self, state):
        # if state == (10,1):
            # print(state, 0 <= state[0] < self.dim[0] and 0 <= state[1] < self.dim[1])
        return 0 <= state[0] < self.dim[0] and 0 <= state[1] < self.dim[1]

        # s_, p = self.T[s][a]
        # if np.random.rand() < p:
        #     return s_, self.R[s_]
        # else:
        #     return s, -10



# class GridWorld():

#     def __init__(self, dim, start, goal, obstacles, rewards, static_obstacles=[], static_rewards=[]):
#         self.dim = dim
#         self.start = start
#         self.current = deepcopy(self.start)
#         self.goal = goal
#         self.obstacles = obstacles # dict mapping obstacles to probabilities
#         self.episodic_obstacles = []
#         self.rewards = rewards
#         self.rewards[self.goal] = 0
#         self.static_rewards = static_rewards
#         self.static_obstacles = static_obstacles
#         self.episodic_rewards = deepcopy(rewards)
#         np.random.seed(42)

#     def sample(self):
#         self.reset()
#         self.episodic_obstacles = []
#         self.episodic_rewards = deepcopy(self.rewards)
#         for obstacle, p in self.obstacles.items():
#             if np.random.random() < p:
#                 self.episodic_obstacles.append(obstacle)
#                 self.episodic_rewards[obstacle] = -10

#     def action(self, action, target=None):
#         if isinstance(action, Action):
#             state = tuple(map(operator.add, self.current, ACTION_MODIFIERS[action]))
#             if self.ok_action(action):
#                 self.current = state
#                 return self.episodic_rewards[self.current], self.current == self.goal
#             else:
#                 return -10, False
#         elif isinstance(action, MetaAction):
#             if action == MetaAction.PLACE_OBJECT:
#                 self.obstacles[target] = 1.0
#                 self.rewards[target[0]][target[1]] = -10
#             elif action == MetaAction.REMOVE_OBJECT:
#                 self.obstacles.pop(target)
#                 self.rewards[target[0]][target[1]] = -2
#             elif action == MetaAction.INCREASE_REWARD:
#                 self.rewards[target[0]][target[1]] = -1
#             elif action == MetaAction.DECREASE_REWARD:
#                 self.rewards[target[0]][target[1]] = -2
#             return 0, False

#     def estimate_reward(self, action):
#         if not self.ok_action(action, episodic=False):
#             return -10
#         else:
#             return self.rewards[tuple(map(operator.add, self.current, ACTION_MODIFIERS[action]))]

#     def reset(self):
#         self.current = deepcopy(self.start)

#     def static(self):
#         self.reset()
#         self.episodic_rewards = self.static_rewards
#         self.episodic_obstacles = self.static_obstacles

#     def ok_action(self, action, episodic=True):
#         return self.ok_state(tuple(map(operator.add, self.current, ACTION_MODIFIERS[action])), episodic=episodic)

#     def ok_state(self, state, episodic=True):
#         if episodic:
#             if 0 <= state[0] < self.dim[0] and 0 <= state[1] < self.dim[1] and not state in self.episodic_obstacles:
#                 return True
#         else:
#             if 0 <= state[0] < self.dim[0] and 0 <= state[1] < self.dim[1] and not state in self.obstacles.keys():
#                 return True # okay for now...
#         return False
    
#     def feasible_action(self, action):
#         return self.feasible_state(tuple(map(operator.add, self.current, ACTION_MODIFIERS[action])))
