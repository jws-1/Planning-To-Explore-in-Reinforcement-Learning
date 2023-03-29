from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
# from ..models.mdp import MetaAction
from ..actions import MetaAction
from collections import defaultdict
class MetaPRLAgent(RLAgent):

    def __init__(self, env, model):
        self.env = env
        self.initial_model = model
        self.reset()

    def reset(self):
        self.Q = {state : {action : 100. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)
        self.N_sa = {state : {action : 0 for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.N_sas = {state : {action : {next_state : 0 for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.meta_actions = []

    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))
        next_action = None
        follows_meta_action = False
        for i in range(config.episodes):
            # if i % (config.episodes // 10) == 0:
            print(f"Meta PRL-AGENT: episode {i}")

            # self.meta_sas = {state : {action : {next_state : {meta_action: False for meta_action in MetaAction} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
    
            # self.meta_sas = {state : {action : {next_state: {ma : False for ma in self.meta_actions} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
            self.meta_sa = {state : {action : {ma : False for ma in self.meta_actions} for action in range(self.env.nA)} for state in range(self.env.nS)}
            # for action in self.meta_actions:
            #     self.meta_sas
            self.observed_sas = {state : {action : {next_state: False for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}

            done = False
            state = self.env.reset()

            planning = i < config.planning_steps

            while not done:
                
                if next_action is not None:
                    action = next_action
                    next_action = None
                    follows_meta_action = True

                elif planning:
                    plan = self.model.plan_VI(state, self.env.goal, meta=True, observed_sas=self.observed_sas, meta_sa=self.meta_sa, meta_actions = self.meta_actions)
                    if isinstance(plan, tuple):
                        if len(plan) == 4:
                            _, target_action, action, target_state = plan
                            # action, target_action, target_state = plan
                            # next_action = target_action
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])
                
                if isinstance(action, MetaAction):
                    self.model.update_transition_prob(state, target_action, target_state, 1.0)
                    self.meta_sa[state][target_action][action] = True
                    next_action = target_action
                    # if action == MetaAction.INCREASE_REWARD:
                    #     # print(state, action, target_action, target_state)
                    #     self.model.update_reward(state, target_action, target_state, np.max(self.model.reward_function))
                    # elif action == MetaAction.INCREASE_TRANSITION_PROBABILITY:
                    #     # print(state, action, target_action, target_state)
                    #     self.model.update_transition_prob(state, target_action, target_state, 1.0)
                    # self.meta_sas[state][target_action][target_state][action] = True
                    
                else:

                    next_state, reward, done, _ = self.env.step(action)
                    # print(np.unravel_index(state, (7,10)), action, np.unravel_index(next_state, (7,10)))
                    # print(state, action, next_state)
                    # self.env.render()
                    old_value = self.Q[state][action]
                    next_max = max(self.Q[next_state].values())
                    new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                    self.Q[state][action] = new_value

                    if planning:
                        
                        if not follows_meta_action:
                            if state != next_state:
                            
                                action_sequence = self.model.action_sequence(state, next_state)
                                # print(action_sequence)
                                if action_sequence != [action]:
                                    meta_action = MetaAction(action, action_sequence)
                                    if not meta_action in self.meta_actions:
                                        self.meta_actions.append(meta_action)
                                        self.meta_sa[state][action][meta_action] = False
                        else:
                            follows_meta_action = False
                        self.N_sa[state][action]+=1
                        self.N_sas[state][action][next_state]+=1
                        self.model.update_transition_probs(state, action, self.N_sa[state][action], self.N_sas[state][action])
                        self.model.update_reward(state, action, next_state, reward)
                        self.observed_sas[state][action][next_state ] = True


                    if state != next_state:
                        states[i][state]+=1

                    state = next_state
                    rewards[i] += reward    

            states[i][state]+=1
        with open("learnt_actions.txt", "w+") as fp:
            fp.write("\n".join([str(m) for m in self.meta_actions]))
        return rewards, states