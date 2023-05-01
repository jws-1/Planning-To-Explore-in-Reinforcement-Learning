# from ..agents import MetaPRLAgent, PRLAgent, RLAgent
# import gym
# import numpy as np
# from ..models import MDP
# from types import SimpleNamespace
# from plot import plot_results
# from collections import defaultdict
# def generate_inaccurate_mdp(env, mdp):
#     return mdp

from ..models import MDP
import gym
from .benchmark_env import BenchmarkEnv
import os
import numpy as np
import matplotlib.pyplot as plt
class BenchmarkFrozenLake(BenchmarkEnv):
    
    def __init__(self):
        self.env_name = "FrozenLake-v1"
        self.env_kwargs = {"is_slippery":True, "map_name" : "4x4"}

    def reset_env(self):
        self.env = gym.make(self.env_name, **self.env_kwargs)
        self.env.seed(442)
        self.env.max_reward = 1.0
        self.env.start = 0

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        """
        Make this inaccurate, in some way.
        """
        T = np.zeros((self.env.nS, self.env.nA, self.env.nS), dtype=int)
        R = np.full((self.env.nS, self.env.nA, self.env.nS), -1., dtype=float)

        reasonable_meta_states = {}

        for s in range(self.env.nS):

            reasonable_meta_states[s] = []

            s2d = np.unravel_index(s, (4,4))
            up2d = s2d + np.array([-1, 0])
            up2d = np.clip(up2d, np.array([0,0]), np.array([3,3]))
            up = np.ravel_multi_index(up2d, (4,4))
            reasonable_meta_states[s].append(up)

            right2d = s2d + np.array([0, 1])
            right2d = np.clip(right2d, np.array([0,0]), np.array([3,3]))
            right = np.ravel_multi_index(right2d, (4,4))
            reasonable_meta_states[s].append(right)

            down2d = s2d + np.array([1, 0])
            down2d = np.clip(down2d, np.array([0,0]), np.array([3,3]))
            down = np.ravel_multi_index(down2d,  (4,4))
            reasonable_meta_states[s].append(down)

            left2d = s2d + np.array([0, -1])
            left2d = np.clip(left2d, np.array([0,0]), np.array([3,3]))
            left = np.ravel_multi_index(left2d, (4,4))
            reasonable_meta_states[s].append(left)

            T[s, 3, up] = 1.0
            T[s, 2, right] = 1.0
            T[s, 1, down] = 1.0
            T[s, 0, left] = 1.0


        model = MDP(np.array(range(self.env.nS)), np.array([15]), np.array(range(self.env.nA)), T, R, reasonable_meta_transitions=None if not reasonable_meta else reasonable_meta_states, planner=planner, undiscretize_fn=lambda x: np.unravel_index(x, (4,4)))
        return model


    def handle_results(self, results, p, w):
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists(os.path.join("results", "stochastic")):
            os.mkdir(os.path.join("results", "stochastic"))
        if not os.path.exists(os.path.join("results", "stochastic", "frozen_lake")):
            os.mkdir(os.path.join("results", "stochastic", "frozen_lake"))
        
        for agent, result in results.items():
            rewards, rewards_95pc, states, states_95pc, regrets, regrets_95pc = result
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-rewards.npy")), rewards)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-rewards_95pc.npy")), rewards_95pc)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-states.npy")), states)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-states_95pc.npy")), states_95pc)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-regrets.npy")), regrets)
            np.save((os.path.join("results", "stochastic", "frozen_lake", f"{agent}-regrets_95pc.npy")), regrets_95pc)




# mb_learn_config_dict = {
#     "m": 5,
#     "episodes": 1000,
#     "window_size":1,
#     "planning_steps":100,
#     "eps": 0.0,
#     "lr": 0.6,
#     "df": 1.0,
#     "learn_model":True,
# }

# mb_config_dict = {
#     "m": 5,
#     "episodes": 50,
#     "window_size":1,
#     "planning_steps":100,
#     "eps": 0.0,
#     "lr": 0.6,
#     "df": 1.0,
#     "learn_model":False,
# }

# mf_config_dict = {
#     "m": 5,
#     "episodes": 1000,
#     "window_size":1,
#     "eps": 0.5,
#     "eps_min": 0.1,
#     "decay": True,
#     "lr": 0.6,
#     "df": 1.0,
# }

# def make_transition_function(height, width):
#     num_states = height * width
#     actions = ["left", "down", "right", "up"]
#     num_actions = len(actions)

#     # Define the transition function as a 3D array
#     transition_func = np.zeros((num_states, num_actions, num_states))

#     # Define the helper functions for state indexing
#     def get_state_index(row, col):
#         return np.ravel_multi_index((row, col), (height, width))

#     def get_state_coords(state_index):
#         return np.unravel_index(state_index, (height, width))

#     # Define the transition probabilities for each action
#     for row in range(height):
#         for col in range(width):
#             state_index = get_state_index(row, col)
#             for action_index, action in enumerate(actions):
#                 if action == "up":
#                     next_row = max(row-1, 0)
#                     next_col = col
#                 elif action == "right":
#                     next_row = row
#                     next_col = min(col+1, width-1)
#                 elif action == "down":
#                     next_row = min(row+1, height-1)
#                     next_col = col
#                 elif action == "left":
#                     next_row = row
#                     next_col = max(col-1, 0)

#                 next_state_index = get_state_index(next_row, next_col)

#                 transition_func[state_index, action_index, next_state_index] = 1.0

#     return transition_func

# def generate_transition_function(height, width):
#     nS = height * width  # total number of states
#     nA = 4  # number of actions (up, right, down, left)
#     P = np.zeros((nS, nA, nS))  # initialize transition probabilities array

#     for s in range(nS):
#         row = s // width  # current row of state s
#         col = s % width  # current column of state s

#         # transition probabilities for action "up"
#         if row > 0:
#             next_s = s - width  # next state if action "up" is taken
#             P[s, 0, next_s] = 1.0

#         # transition probabilities for action "right"
#         if col < width - 1:
#             next_s = s + 1  # next state if action "right" is taken
#             P[s, 1, next_s] = 1.0

#         # transition probabilities for action "down"
#         if row < height - 1:
#             next_s = s + width  # next state if action "down" is taken
#             P[s, 2, next_s] = 1.0

#         # transition probabilities for action "left"
#         if col > 0:
#             next_s = s - 1  # next state if action "left" is taken
#             P[s, 3, next_s] = 1.0

#     return P

# def generate_reasonable_transitions(env):
#     reasonable = defaultdict(list)

#     for state in range(env.nS):
#         state_2d =  np.unravel_index(state, (env.grid_height, env.grid_width))
#         if state_2d[0]+1 < env.grid_height:
#             reasonable[state].append(np.ravel_multi_index((state_2d[0]+1, state_2d[1]),  (env.grid_height, env.grid_width)))
#         if state_2d[0]-1 >= 0:
#             reasonable[state].append(np.ravel_multi_index((state_2d[0]-1, state_2d[1]),  (env.grid_height, env.grid_width)))
#         if state_2d[1]+1 < env.grid_width:
#             reasonable[state].append(np.ravel_multi_index((state_2d[0], state_2d[1]+1),  (env.grid_height, env.grid_width)))
#         if state_2d[1]-1 >= 0:
#             reasonable[state].append(np.ravel_multi_index((state_2d[0], state_2d[1]-1),  (env.grid_height, env.grid_width)))
    
#     return reasonable

# def benchmark(agent_cls, learn=True):
#     np.random.seed(42)
#     env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
#     env.seed(42)
#     T = np.zeros((env.nS, env.nA, env.nS), dtype=np.int64)
#     R = np.zeros((env.nS, env.nA, env.nS), dtype=np.float64)
#     # print(env.nA)
#     # for s in range(env.nS):
#     #     if s == 13:
#     #     for a in range(env.nA):
#     #         for next_s in range(env.nS):
#     #             # T[s][a][next_s] = env.P[s][a][next_s]
#     #             R[s][a][next_s] = -2.
#     # R[:, :, :] = -2.
#     # R[:, :, :] = -2
#     env.grid_height = env.grid_width = 4
#     env.goal = 15
#     T = make_transition_function(env.grid_height, env.grid_width)
#     R[:, :, env.goal] = 1.0
#     # R[:, :, 37] = 1.0   
#     # T[30, 1, 31] = 0.8
#     # T[30, 1, 30] = 0.2
#     # print(R[38, 3, 37])
#             # for p, next_s, r, _ in env.P[s][a]:
#             #     T[s][a][next_s] = p
#             #     R[s][a][next_s] = r
#     # print(T[31])
#     reasonable = generate_reasonable_transitions(env)
#     # print(reasonable)
#     mdp = MDP(states=np.array(range(env.nS)),
#                 goal=env.goal,
#                 actions=np.array(range(env.nA)),
#                 transition_function=T,
#                 reward_function=R,
#                 discount_factor=1.0, reasonable_meta_transitions=reasonable)
    
#     if agent_cls in [MetaPRLAgent, PRLAgent]:
#         inaccurate_mdp = generate_inaccurate_mdp(env, mdp)
#         agent = agent_cls(env, inaccurate_mdp)
#         if learn:
#             config = SimpleNamespace(**mb_learn_config_dict)
#         else:
#             config = SimpleNamespace(**mb_config_dict)
#     else:
#         agent = agent_cls(env)
#         config = SimpleNamespace(**mf_config_dict)
    
#     rewards, rewards_95pc, states = agent.learn_and_aggregate(config)

#     return rewards, rewards_95pc, states

# if __name__ == "__main__":
#     results = {
#         "MetaPRL" : benchmark(MetaPRLAgent),
#         "PRL" : benchmark(PRLAgent),
#         "RL" : benchmark(RLAgent)
#     }
#     plot_results(results, "stochastic_transition_results", optimal_reward=-15.)
#     # plot_states_heatmap(results)
#     for agent, (rewards, rewards_95pc, states) in results.items():
#         if agent in ["MetaPRL", "PRL", "DumbPRL"]:
#             print(f"{agent} min, max, mean, final planning, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[mb_learn_config_dict['planning_steps']-mb_learn_config_dict['window_size']], rewards[-1]}")
#         else:
#             print(f"{agent} min, max, mean, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[-1]}")
    