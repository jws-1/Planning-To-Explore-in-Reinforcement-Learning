from ..agents import MetaPRLAgent
import gym
import gym_windy_gridworlds
import numpy as np
from ..models import MDP
from types import SimpleNamespace
from plot import plot_results

def generate_inaccurate_mdp(env, mdp):
    return mdp

mb_learn_config_dict = {
    "m": 5,
    "episodes": 100,
    "window_size":1,
    "planning_steps":20,
    "eps": 0.0,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":True,
}

mb_config_dict = {
    "m": 5,
    "episodes": 100,
    "window_size":10,
    "planning_steps":20,
    "eps": 0.1,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":False,
}

mf_config_dict = {
    "m": 5,
    "episodes": 100,
    "window_size":10,
    "eps": 0.5,
    "eps_min": 0.1,
    "decay": False,
    "lr": 0.6,
    "df": 1.0,
}

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


def benchmark(agent_cls, learn=True):
    np.random.seed(42)
    env = gym.make("StochWindyGridWorld-v0")
    T = np.zeros((env.nS, env.nA, env.nS), dtype=np.int64)
    R = np.zeros((env.nS, env.nA, env.nS), dtype=np.float64)
    print(env.nA)
    for s in range(env.nS):
        if s == 13:
            print("12")
        for a in range(env.nA):
            for next_s in range(env.nS):
                # T[s][a][next_s] = env.P[s][a][next_s]
                R[s][a][next_s] = -2.
    T = generate_transition_function(env.grid_height, env.grid_width)
    R[:, :, env.goal_state] = 1.0
            # for p, next_s, r, _ in env.P[s][a]:
            #     T[s][a][next_s] = p
            #     R[s][a][next_s] = r
    print(T[13])
    print(env.P[13])


    mdp = MDP(states=np.array(range(env.nS)),
                actions=np.array(range(env.nA)),
                transition_function=env.P,
                reward_function=R,
                discount_factor=1.0)
    
    if agent_cls in [MetaPRLAgent]:
        inaccurate_mdp = generate_inaccurate_mdp(env, mdp)
        agent = agent_cls(env, inaccurate_mdp)
        if learn:
            config = SimpleNamespace(**mb_learn_config_dict)
        else:
            config = SimpleNamespace(**mb_config_dict)
    else:
        agent = agent_cls(env)
        config = SimpleNamespace(**mf_config_dict)
    
    rewards, rewards_95pc, states = agent.learn_and_aggregate(config)

    return rewards, rewards_95pc, states

if __name__ == "__main__":
    results = {
        "MetaPRL" : benchmark(MetaPRLAgent),
    }
    plot_results(results, "stochastic_transition_results", optimal_reward=-18.)
    # plot_states_heatmap(results)
    for agent, (rewards, rewards_95pc, states) in results.items():
        if agent in ["MetaPRL", "PRL", "DumbPRL"]:
            print(f"{agent} min, max, mean, final planning, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[mb_learn_config_dict['planning_steps']-mb_learn_config_dict['window_size']], rewards[-1]}")
        else:
            print(f"{agent} min, max, mean, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[-1]}")
    