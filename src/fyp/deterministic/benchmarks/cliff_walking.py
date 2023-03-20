from ..agents import PRLAgent, MetaPRLAgent, RLAgent

from ..models.deterministic_mdp import D_MDP
import gym
import numpy as np
from types import SimpleNamespace
from plot import plot_results


def generate_inaccurate_mdp(env, mdp, p_tf, p_r):
    new_tf = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int64)
    new_rf = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    for state in mdp.states:
        for action in mdp.actions:
            next_state = mdp.get_transition(state, action)
            reward = mdp.get_reward(state, action)
            if np.random.random() < p_tf:
                # Choose a random next state
                new_tf[state, action] = np.random.choice(list(mdp.states))
            else:
                # Use the original transition function
                new_tf[state, action] = next_state
            if next_state == env.goal:
                new_tf[state, action] = next_state
            if np.random.random() < p_r and mdp.transition_function[state, action] != env.goal:
            #     # Choose a random reward
                new_rf[state, action] = np.random.uniform(-2., -1.)
            else:
                # Use the original reward function
                new_rf[state, action] = reward
            if next_state == env.goal:
                new_rf[state, action] = 10.
    return D_MDP(mdp.states, mdp.actions, new_tf, new_rf, mdp.discount_factor)


mb_learn_config_dict = {
    "m": 5,
    "episodes": 100,
    "window_size":10,
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
    "eps": 0.0,
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

def benchmark(agent_cls, learn=True):
    np.random.seed(42)
    env = gym.make("CliffWalking-v0")
    env.goal = 47
    T = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int64)
    R = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    # T = {}
    # R = {}
    for s in range(env.observation_space.n):
    #     T[s] = {}
    #     R[s] = {}
        for a in range(env.action_space.n):
            for p, next_s, r, _ in env.P[s][a]:
                if p == 1.0:
                    T[s, a] = next_s
                    R[s, a] = r
    #                 T[s][a] = next_s
    #                 R[s][a] = r

    mdp = D_MDP(states=np.array(range(env.observation_space.n)),
                actions=np.array(range(env.action_space.n)),
                transition_function=T,
                reward_function=R,#lambda s,a: env.P[s][a][0][2],
                discount_factor=1.0)
    
    if agent_cls in [MetaPRLAgent, PRLAgent]:
        inaccurate_mdp = generate_inaccurate_mdp(env, mdp, 0.5, 0.5)
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


def plot_states_heatmap(results, dir):
    pass


if __name__ == "__main__":
    results = {
        "MetaPRL" : benchmark(MetaPRLAgent),
        "PRL" : benchmark(PRLAgent),
        # "DumbPRL" : benchmark(PRLAgent, False),
        "RL" : benchmark(RLAgent)
    }
    plot_results(results, "deterministic_results", optimal_reward=-13.)
    # plot_states_heatmap(results)
    for agent, (rewards, rewards_95pc, states) in results.items():
        if agent in ["MetaPRL", "PRL", "DumbPRL"]:
            print(f"{agent} min, max, mean, final planning, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[mb_learn_config_dict['planning_steps']-mb_learn_config_dict['window_size']], rewards[-1]}")
        else:
            print(f"{agent} min, max, mean, final model-free rewards: {min(rewards), max(rewards), np.mean(rewards), rewards[-1]}")
    