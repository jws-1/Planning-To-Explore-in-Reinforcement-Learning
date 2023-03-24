from ..agents import UCB, MetaUCB
from plot import plot_result_bandit
from ..envs import KArmedBandit
from types import SimpleNamespace
import numpy as np
ucb_config_dict = {
    "m": 1,
    "max_actions": 100,
    "window_size": 20,
    "c": 2,
    "planning_steps":20
}

def generate_q_star(k):
    """Generate true action values for a k-armed bandit with Gaussian reward distributions."""
    q_star = np.zeros(k)
    for i in range(k):
        q_star[i] = np.random.normal(loc=0, scale=1)
    return q_star

q_star = generate_q_star(10)
mean_rewards = np.zeros(10)
for i in range(10):
    mean_rewards[i] = np.mean(np.random.normal(loc=q_star[i], scale=1, size=10000))
print("Mean reward for each arm:", mean_rewards)


def benchmark(agent_cls):

    env = KArmedBandit(10, seed=69, q_star=q_star)
    agent = agent_cls(env)
    rewards, rewards_95pc, actions = agent.learn_and_aggregate(SimpleNamespace(**ucb_config_dict))
    return rewards, rewards_95pc, actions

if __name__ == "__main__":
    results = {
        "UCB": benchmark(UCB),
        "MetaUCB" : benchmark(MetaUCB)
    }
    plot_result_bandit(results, "stochastic_bandit_results")
    for agent, (rewards, rewards_95pc, arms) in results.items():
        print(f"{agent} min, max, mean, final rewards: {np.min(rewards), np.max(rewards), np.mean(rewards), rewards[-1]}")
