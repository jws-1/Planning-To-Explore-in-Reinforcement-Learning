
from ..agents import MetaPRLAgent, PRLAgent, RLAgent
import numpy as np
from types import SimpleNamespace

meta_prl_config_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "planning_steps":20,
    "lr": 0.6,
    "df": 1.0,
    "learn_meta_actions":False
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
    "eps" : 0.0
}


mf_config_dc_dict = {
    "m": 10,
    "episodes": 200,
    "window_size":10,
    "eps": 1.0,
    "eps_min": 0.01,
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

    def generate_model(self, reasonable_meta=False, noise=False, planner="VI"):
        pass

    def generate_reasonable_meta(self, env):
        pass

    def handle_results(self, results):
        pass

    def reset_env(self):
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)


    def perform_benchmarks(self, m=10, n=1000, p=100, w = 20):
        results = {}

        for config in configs:
            config["m"] = m
            config["episodes"] = n
            if hasattr(config, "planning_steps"):
                config["planning_steps"] = p
            config["window_size"] = w


        # self.reset_env()
        # meta_reasonable_model = self.generate_model(False, True, planner="A*")
        # meta_reasonable = MetaPRLAgent(self.env, meta_reasonable_model)
        # results["RL_AStar_Meta_Learn"] = meta_reasonable.learn_and_aggregate(SimpleNamespace(**meta_prl_learn_config_dict))
        # print("RL_AStar_Learn done")

        self.reset_env()
        meta_reasonable_model = self.generate_model(True, True, planner="A*")
        meta_reasonable = MetaPRLAgent(self.env, meta_reasonable_model)
        results["RL_AStar_Meta_Reasonable"] = meta_reasonable.learn_and_aggregate(SimpleNamespace(**meta_prl_config_dict))
        print("RL_AStar_Reasonable done")


        # MetaPRL with embedded reasonable meta actions.
        self.reset_env()
        meta_reasonable_model = self.generate_model(True, True)
        meta_reasonable = MetaPRLAgent(self.env, meta_reasonable_model)
        results["RL_VI_Meta_Reasonable"] = meta_reasonable.learn_and_aggregate(SimpleNamespace(**meta_prl_config_dict))
        print("RL_VI_Reasonable done")


        # # MetaPRL with learning meta actions
        # self.reset_env()
        # meta_model = self.generate_model(False, True)
        # meta = MetaPRLAgent(self.env, meta_model)
        # results["RL_VI_Meta_Learn"] = meta.learn_and_aggregate(SimpleNamespace(**meta_prl_learn_config_dict))

        # PRL
        self.reset_env()
        prl_model = self.generate_model(False, True)
        prl = PRLAgent(self.env, prl_model)
        results["PRL"] = prl.learn_and_aggregate(SimpleNamespace(**prl_learn_config_dict))
        print("PRL done")

        # RL
        self.reset_env()
        rl = RLAgent(self.env)
        results["RL"] = rl.learn_and_aggregate(SimpleNamespace(**mf_config_dc_dict))
        print("RL done")

        return self.handle_results(results, p, w)