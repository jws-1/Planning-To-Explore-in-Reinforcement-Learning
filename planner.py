from collections import defaultdict
import heapq
import operator
from actions import Action, ACTION_MODIFIERS
from copy import deepcopy

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

class Planner():
    
    def __init__(self, model):
        self.model = model

    def plan(self):
        return self.a_star()[0]
    
    def actions_from_path(self, path):
        actions = [] # list of actions
        rev_modifiers = dict((v,k) for k,v in ACTION_MODIFIERS.items())
        for curr, next in zip(path, path[1:]):
            diff = tuple(map(operator.sub, next, curr))
            actions.append(rev_modifiers[diff])
        return actions
    

    def a_star(self):
        W = 1
        start = deepcopy(self.model.current)
        goal = tuple(self.model.goal)
        g = defaultdict(
            lambda: float("inf")
        )  # dictionary with key: (x,y) tuple and value: length of the shortest path from (0,0) to (x,y)
        g[tuple(start)] = 0 # set distance to source 0
        parent = {
            tuple(start): None
        }  # dictionary with key: (x,y) tuple and value is a (x,y) tuple where the value point is the parent point of the key point in terms of path traversal
        pq = [(0, -g[tuple(start)], tuple(start))]
        trajectory = cells_processed = 0 # statistics to keep track while A* runs

        while pq:
            _, _, curr = heapq.heappop(pq)  # pop from priority queue
            # pretty_print_explored_region(grid, list(g.keys())) # uncomment if you want to see each step of A*
            trajectory += 1
            if curr == goal:  # check if we found our goal point
                break

            # generate children
            for dx, dy in [(0, 1), (-1, 0), (0, -1), (1, 0)]:  # up left down right
                new_x, new_y = (curr[0] + dx, curr[1] + dy)  # compute children

                children = (new_x, new_y)
                if not self.model.feasible_state(children):
                    continue
                # # print(self.model.obstacles)
                if children in self.model.obstacles.keys():
                    continue
                # print(self.model.obstacles)
                # if children in self.model.obstacles.keys():
                #     if self.model.obstacles[children] > 0.5:
                #         continue
                #     d = self.model.obstacles[children]# + (1-self.model.obstacles[children])
                #     # print(d)
                # else:
                #     d = 1
                cost = abs(self.model.rewards[children])
                new_g= g[curr] + cost # add + 1 to the real cost of children
                if (
                    children not in g or new_g < g[children]
                ):  # only care about new undiscovered children or children with lower cost
                    cells_processed += 1
                    g[children] = new_g  # update cost
                    parent[children] = curr  # update parent of children to current
                    h_value = manhattan_distance(*children, *goal) # calculate h value
                    f = new_g + W * h_value  # generate f(n')
                    heapq.heappush(
                        pq, (f, -new_g, children)
                    )  # add children to priority queue
        else:
            return []

        # generate path traversal using parent dict
        path = [curr]
        while curr != tuple(start):
            curr = parent[curr]
            path.append(curr)
        path.reverse()  # reverse the path so that it is the right order
        return self.actions_from_path(path)