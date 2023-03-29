from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
from ..actions import MetaAction

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
                    plan = self.model.plan_VI(state, self.env.goal, meta=True, observed_sa=self.observed_sa, meta_sa=self.meta_sa, meta_actions = self.meta_actions)
                    if isinstance(plan, tuple):
                        _, target_action, action, target_state = plan
                        # if len(plan) == 3:
                        #     action, target_action, target_state = plan
                        #     next_action = target_action
                        # elif len(plan) == 2:
                        #     action, target_action = plan
                        #     next_action = target_action
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])

                if isinstance(action, MetaAction):
                    self.model.update_transition(state, target_action, target_state)
                    self.meta_sa[state][target_action][action] = True
                    # if action == MetaAction.INCREASE_REWARD:
                    #     print(state, action, target_action)
                    #     self.model.update_reward(state, target_action, self.model.get_reward(state, target_action)+1.0)
                    # elif action == MetaAction.ADD_TRANSITION:
                    #     print(state, action, target_action, target_state)
                    #     self.model.update_transition(state, target_action, target_state)
                    # self.meta_sa[state][target_action][action] = True
                else:
                    next_state, reward, done, _ = self.env.step(action)
                    print(state, action, next_state)
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