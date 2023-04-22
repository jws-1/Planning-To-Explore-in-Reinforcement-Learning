from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
from ..actions import BaseMetaActions

class MetaPRLAgent(RLAgent):

    def __init__(self, env, model):
        self.env = env
        self.initial_model = model
        self.reset()

    def reset(self):
        self.meta_actions = []
        self.Q = {state : {action : 100. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)
        self.observed_sa = {state : {action : False for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.meta_sa = {state : {action : {meta_action : False for meta_action in self.meta_actions} for action in range(self.env.nA)} for state in range(self.env.nS)}
        # self.meta_sas = {state : {action : {next_state : {meta_action: False for meta_action in MetaAction} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
 
    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))
        next_action = None
        for i in range(config.episodes):
            # if i % (config.episodes // 10) == 0:
            print(f"Meta PRL-AGENT: episode {i}")

            done = False
            state = self.env.reset()

            planning = i < config.planning_steps
            follows_meta_action = False
            while not done:
                
                if next_action is not None:
                    action = next_action
                    next_action = None

                elif planning:
                    plan = self.model.plan_VI(state, meta=True, observed_sa=self.observed_sa, meta_sa=self.meta_sa)
                    if isinstance(plan, tuple):
                        action, target_action, target_state = plan
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])

                if isinstance(action, BaseMetaActions):
                    if action == BaseMetaActions.ADD_TRANSITION:
                        self.model.update_transition(state, target_action, target_state)
                    elif action == BaseMetaActions.INCREASE_REWARD:
                        self.model.update_reward(state, target_action, self.model.reward_function[state, action]+1)
                    self.meta_sa[state][target_action][action] = True
                else:
                    next_state, reward, done, _ = self.env.step(action)
                    self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])


                    if planning:
                        self.model.update_transition(state, action, next_state)
                        self.model.update_reward(state, action, reward)
                        self.observed_sa[state][action] = True

                    if state != next_state:
                        states[i][state]+=1
                
                    state = next_state
                    rewards[i] += reward    


            states[i][state]+=1
        with open("learnt_actions.txt", "w+") as fp:
            fp.write("\n".join([str(m) for m in self.meta_actions]))
        return rewards, states