from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
from ..models.mdp import MetaAction

class MetaPRLAgent(RLAgent):

    def __init__(self, env, model):
        self.env = env
        self.initial_model = model
        self.reset()

    def reset(self):
        self.Q = {state : {action : 0. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)
        self.N_sa = {state : {action : 0 for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.N_sas = {state : {action : {next_state : 0 for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.meta_sas = {state : {action : {next_state: {meta_action : False for meta_action in MetaAction} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
        print("Reset!")
        print(self.model.transition_function)

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))
        expected_next_state = next_action = prev_reward = None
        self.observed_sas = {state : {action : {next_state: False for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}

        for i in range(config.episodes):
            # if i % (config.episodes // 10) == 0:
            print(f"Meta PRL-AGENT: episode {i}")

            # self.meta_sas = {state : {action : {next_state : {meta_action: False for meta_action in MetaAction} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
    

            done = False
            state = self.env.reset()
            planning = i < config.planning_steps

            while not done:
                # print(self.model.transition_function[state])
                if next_action is not None:
                    action = next_action
                    next_action = None
                elif planning:
                    plan = self.model.plan_VI(state, meta=True, observed_sas=self.observed_sas, meta_sas=self.meta_sas)
                    if isinstance(plan, tuple):
                        if len(plan) == 3:
                            action, target_action, target_state = plan
                            next_action = target_action
                            expected_next_state = target_state
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])

                if isinstance(action, MetaAction):
                    if action == MetaAction.INCREASE_REWARD:
                        # print(state, action, target_action, target_state)
                        prev_reward = self.model.reward_function[state, target_action, target_state]
                        self.model.update_reward(state, target_action, target_state, np.max(self.model.reward_function[state, target_action, target_state])-1)
                    elif action == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                        # print(state, action, target_action, target_state)
                        self.model.update_transition_prob(state, target_action, target_state, 1.0)
                    self.meta_sas[state][target_action][target_state][action] = True
                else:
                    next_state, reward, done, info = self.env.step(action)
                    if done and not info.get("TimeLimit.truncated"):
                        print("Completed ", i)
                    # print(state, action, next_state)
                    self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])

                    if planning:
                        self.N_sa[state][action]+=1
                        self.N_sas[state][action][next_state]+=1
                        self.model.update_transition_probs(state, action, self.N_sa[state][action], self.N_sas[state][action])
                        self.model.update_reward(state, action, next_state, reward)
                        self.observed_sas[state][action][next_state ] = True

                    if expected_next_state is not None and prev_reward is not None:
                        if next_state != expected_next_state:
                            self.model.update_reward(state, action, expected_next_state, prev_reward)
                            prev_reward = None
                            expected_next_state = None

                    if state != next_state:
                        states[i][state]+=1
                
                    state = next_state
                    rewards[i] += reward    

            states[i][state]+=1
        print(rewards)
        return rewards, states