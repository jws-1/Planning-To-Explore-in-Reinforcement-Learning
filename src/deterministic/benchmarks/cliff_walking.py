from deterministic.agents import PRLAgent, MetaPRLAgent, RLAgent

from models.deterministic_mdp import D_MDP
import gym
import numpy as np
from types import SimpleNamespace
from plot import plot_results


def generate_inaccurate_mdp(mdp, p_tf, p_r):
    new_tf = {}
    new_rf = {}
    for state in mdp.states:
        new_tf[state] = {}
        new_rf[state] = {}
        for action in mdp.actions:
            next_state = mdp.get_transition(state, action)
            reward = mdp.get_reward(state, action)
            if np.random.random() < p_tf:
                # Choose a random next state
                new_tf[state][action] = np.random.choice(list(mdp.states))
            else:
                # Use the original transition function
                new_tf[state][action] = next_state
            if np.random.random() < p_r:
                # Choose a random reward
                new_rf[state][action] = np.random.normal(loc=0, scale=1)
            else:
                # Use the original reward function
                new_rf[state][action] = reward

    return D_MDP(mdp.states, mdp.actions, new_tf, new_rf, mdp.discount_factor)


mb_learn_config_dict = {
    "m": 20,
    "episodes": 1000,
    "window_size":20,
    "planning_steps":10,
    "eps": 0.5,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":True,
}

mb_config_dict = {
    "m": 20,
    "episodes": 1000,
    "window_size":20,
    "planning_steps":10,
    "eps": 0.5,
    "lr": 0.6,
    "df": 1.0,
    "learn_model":False,
}

mf_config_dict = {
    "m": 20,
    "episodes": 1000,
    "window_size":20,
    "eps": 0.5,
    "lr": 0.6,
    "df": 1.0,
}

mb_config = SimpleNamespace()

def benchmark(agent_cls, learn=True):
    env = gym.make("CliffWalking-v0")

    mdp = D_MDP(states=range(env.observation_space.n),
                actions=range(env.action_space.n),
                transition_function=env.P,
                reward_function=lambda s,a: env.P[s][a][0][2],
                discount_factor=1.0)
    
    if agent_cls in [MetaPRLAgent, PRLAgent]:
        inaccurate_mdp = generate_inaccurate_mdp(mdp, 0.5, 0.5)
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
    np.random.seed(42)
    results = {
        "MetaPRL" : benchmark(MetaPRLAgent),
        "PRL" : benchmark(PRLAgent),
        "DumbPRL" : benchmark(PRLAgent, False),
        "RL" : benchmark(RLAgent)
    }
    plot_results(results, "deterministic_results")