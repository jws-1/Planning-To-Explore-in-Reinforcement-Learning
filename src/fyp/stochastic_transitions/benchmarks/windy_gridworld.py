from ..agents import MetaPRLAgent, PRLAgent, RLAgent
import gym
import gym_windy_gridworlds
import numpy as np
from ..models import MDP
from types import SimpleNamespace
from plot import plot_results
from collections import defaultdict
def generate_inaccurate_mdp(env, mdp):
    return mdp

mb_learn_dc_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "eps": 0.0,
    "lr": 1.0,
    "min_lr": 0.1,
    "decay_lr": True,
    "df": 1.0,
    "learn_model":True,
    "learn_meta_actions" : True
}

mb_learn_ma_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "eps": 0.0,
    "lr": 0.6,
    "df": 1.0,
    "decay_lr": False,
    "learn_model":True,
    "learn_meta_actions" : True
}

mb_learn_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "eps": 0.0,
    "lr": 0.6,
    "df": 1.0,
    "decay_lr": False,
    "learn_model":True,
    "learn_meta_actions" : False
}

mb_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":5,
    "planning_steps":20,
    "eps": 0.0,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":False,
}

mf_config_dc_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "eps": 1.0,
    "eps_min": 0.1,
    "decay": True,
    "lr": 0.6,
    "df": 1.0,
}

mf_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":5,
    "eps": 0.5,
    "eps_min": 0.1,
    "decay": False,
    "lr": 0.6,
    "df": 1.0,
}

def make_transition_function(height, width):
    num_states = height * width
    actions = ["up", "right", "down", "left"]
    num_actions = len(actions)

    # Define the transition function as a 3D array
    transition_func = np.zeros((num_states, num_actions, num_states))

    # Define the helper functions for state indexing
    def get_state_index(row, col):
        return np.ravel_multi_index((row, col), (height, width))

    def get_state_coords(state_index):
        return np.unravel_index(state_index, (height, width))

    # Define the transition probabilities for each action
    for row in range(height):
        for col in range(width):
            state_index = get_state_index(row, col)
            for action_index, action in enumerate(actions):
                if action == "up":
                    next_row = max(row-1, 0)
                    next_col = col
                elif action == "right":
                    next_row = row
                    next_col = min(col+1, width-1)
                elif action == "down":
                    next_row = min(row+1, height-1)
                    next_col = col
                elif action == "left":
                    next_row = row
                    next_col = max(col-1, 0)

                next_state_index = get_state_index(next_row, next_col)

                transition_func[state_index, action_index, next_state_index] = 1.0

    return transition_func

def generate_transition_function(height, width):
    nS = height * width  # total number of states
    nA = 4  # number of actions (up, right, down, left)
    P = np.zeros((nS, nA, nS))  # initialize transition probabilities array

    for s in range(nS):
        row = s // width  # current row of state s
        col = s % width  # current column of state s

        # transition probabilities for action "up"
        if row > 0:
            next_s = s - width  # next state if action "up" is taken
            P[s, 0, next_s] = 1.0

        # transition probabilities for action "right"
        if col < width - 1:
            next_s = s + 1  # next state if action "right" is taken
            P[s, 1, next_s] = 1.0

        # transition probabilities for action "down"
        if row < height - 1:
            next_s = s + width  # next state if action "down" is taken
            P[s, 2, next_s] = 1.0

        # transition probabilities for action "left"
        if col > 0:
            next_s = s - 1  # next state if action "left" is taken
            P[s, 3, next_s] = 1.0

    return P

def generate_reasonable_transitions(env):
    reasonable = defaultdict(list)

    for state in range(env.nS):
        state_2d =  np.unravel_index(state, (env.grid_height, env.grid_width))
        if state_2d[0]+1 < env.grid_height:
            reasonable[state].append(np.ravel_multi_index((state_2d[0]+1, state_2d[1]),  (env.grid_height, env.grid_width)))
        if state_2d[0]-1 >= 0:
            reasonable[state].append(np.ravel_multi_index((state_2d[0]-1, state_2d[1]),  (env.grid_height, env.grid_width)))
        if state_2d[1]+1 < env.grid_width:
            reasonable[state].append(np.ravel_multi_index((state_2d[0], state_2d[1]+1),  (env.grid_height, env.grid_width)))
        if state_2d[1]-1 >= 0:
            reasonable[state].append(np.ravel_multi_index((state_2d[0], state_2d[1]-1),  (env.grid_height, env.grid_width)))
    
    return reasonable

def benchmark(agent_cls, learn=True, decay=False,learn_ma=False):
    np.random.seed(56)
    env = gym.make("StochWindyGridWorld-v0")
    env.seed(56)
    T = np.zeros((env.nS, env.nA, env.nS), dtype=np.int64)
    R = np.zeros((env.nS, env.nA, env.nS), dtype=np.float64)
    # print(env.nA)
    # for s in range(env.nS):
    #     if s == 13:
    #     for a in range(env.nA):
    #         for next_s in range(env.nS):
    #             # T[s][a][next_s] = env.P[s][a][next_s]
    #             R[s][a][next_s] = -2.
    R[:, :, :] = -2.
    env.goal = 37
    T = make_transition_function(env.grid_height, env.grid_width)
    R[:, :, 37] = 1.0
    # T[30, 1, 31] = 0.8
    # T[30, 1, 30] = 0.2
    # print(R[38, 3, 37])
            # for p, next_s, r, _ in env.P[s][a]:
            #     T[s][a][next_s] = p
            #     R[s][a][next_s] = r
    # print(T[31])
    reasonable = generate_reasonable_transitions(env)
    # print(reasonable)
    if learn_ma:
        mdp = MDP(states=np.array(range(env.nS)),
                    goal_states=np.array([env.goal]),
                    actions=np.array(range(env.nA)),
                    transition_function=T,
                    reward_function=R,
                    discount_factor=1.0,) #reasonable_meta_transitions=reasonable)
    else:
        mdp = MDP(states=np.array(range(env.nS)),
                    goal_states=np.array([env.goal]),
                    actions=np.array(range(env.nA)),
                    transition_function=T,
                    reward_function=R,
                    discount_factor=1.0, reasonable_meta_transitions=reasonable)
        
    if agent_cls in [MetaPRLAgent, PRLAgent]:
        inaccurate_mdp = generate_inaccurate_mdp(env, mdp)
        agent = agent_cls(env, inaccurate_mdp)
        if learn:
            if learn_ma:
                config = SimpleNamespace(**mb_learn_ma_config_dict)
            else:
                config = SimpleNamespace(**mb_learn_config_dict)
            # if decay:
            #     config = SimpleNamespace(**mb_learn_dc_config_dict)
            # else:
            #     config = SimpleNamespace(**mb_learn_config_dict)
        else:
            config = SimpleNamespace(**mb_config_dict)
    else:
        if decay:
            config = SimpleNamespace(**mf_config_dc_dict)
        else:
            config = SimpleNamespace(**mf_config_dict)
        agent = agent_cls(env)
    
    rewards, rewards_95pc, states = agent.learn_and_aggregate(config)

    return rewards, rewards_95pc, states

if __name__ == "__main__":
    results = {
        # "MetaPRL_DC" : benchmark(MetaPRLAgent, True, True),

        # "MetaPRL_learn" : benchmark(MetaPRLAgent, True, False, True),
        "MetaPRL" : benchmark(MetaPRLAgent, True, False),
        # "PRL_DC" : benchmark(PRLAgent, True, True),
        "PRL" : benchmark(PRLAgent, True, False),
        "RL_decay" : benchmark(RLAgent, False, True),
        # "RL" : benchmark(RLAgent, False, False)
    }
    plot_results(results, "stochastic_transition_results", optimal_reward=-15.)
    # plot_states_heatmap(results)
    for agent, (rewards, rewards_95pc, states) in results.items():
        np.save(f"{agent}_rewards", np.array(rewards))
        np.save(f"{agent}_rewards95pc", np.array(rewards_95pc))
        np.save(f"{agent}_states", np.array(states))
        if agent in ["MetaPRL", "MetaPRL_DC", "PRL", "DumbPRL", "PRL_DC", "MetaPRL_learn"]:
            print(f"{agent} min, max, mean, final planning, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[mb_learn_config_dict['planning_steps']-mb_learn_config_dict['window_size']], rewards[-1]}")
        else:
            print(f"{agent} min, max, mean, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[-1]}")