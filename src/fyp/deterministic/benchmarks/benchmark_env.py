
from ..agents import MetaPRLAgent, PRLAgent, RLAgent
import numpy as np
meta_prl_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "lr": 0.6,
    "df": 1.0,
}

meta_prl_learn_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "lr": 0.6,
    "df": 1.0,
    "learn_meta_actions" : True   
}

prl_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "lr": 0.6,
    "df": 1.0,
    "eps":0.0
}

prl_learn_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "lr": 0.6,
    "df": 1.0,
    "learn_model" : True,
    "eps" : 0.5
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


configs = [
    meta_prl_config_dict,
    meta_prl_learn_config_dict,
    prl_config_dict,
    prl_learn_config_dict,
    mf_config_dc_dict,
    mf_config_dict
]

import gym

class BenchmarkEnv():

    def __init__(self, seed=42):
        pass

    def generate_model(self, reasonable_meta=False, noise=False):
        pass

    def generate_reasonable_meta(self, env):
        pass

    def handle_results(self, results):
        pass

    def reset_env(self):
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)


    def perform_benchmarks(self, m=10, n=1000, p=100):
        results = {}

        for config in configs:
            config["m"] = m
            config["episodes"] = n
            if hasattr(config, "planning_steps"):
                config["planning_steps"] = p

        # MetaPRL with embedded reasonable meta actions.
        self.reset_env()
        meta_reasonable_model = self.generate_model(True, True)
        meta_reasonable = MetaPRLAgent(self.env, meta_reasonable_model)
        results["MetaPRL_reasonable"] = meta_reasonable.learn_and_aggregate(**meta_prl_config_dict)

        # MetaPRL with learning meta actions
        self.reset_env()
        meta_model = self.generate_model(False, True)
        meta = MetaPRLAgent(self.env, meta_model)
        results["MetaPRL_learn"] = meta.learn_and_aggregate(**meta_prl_learn_config_dict)

        # PRL
        self.reset_env()
        prl_model = self.generate_model(False, True)
        prl = PRLAgent(self.env, prl_model)
        results["PRL"] = prl.learn_and_aggregate(**prl_learn_config_dict)

        # RL
        self.reset_env()
        rl = RLAgent(self.env)
        results["RL"] = rl.learn_and_aggregate(**mf_config_dc_dict)

        return self.handle_results(results)
