from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
# from ..models.mdp import MetaAction
from ..actions import MetaAction, BaseMetaActions, MetaActionT, MetaActionR
from collections import defaultdict
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
        self.meta_actions_r = []
        self.meta_actions_t = []
        # self.meta_sas = {state : {action : {next_state: {meta_action : False for meta_action in BaseMetaActions} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
        print("Reset!")
        print(self.model.transition_function)

    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))
        next_action = None
        follows_meta_action = False
        expected_next_state = next_action = prev_reward = target_reward = target_state = None

        for i in range(config.episodes):
            self.meta_sas = {state : {action : {next_state: {meta_action : False for meta_action in BaseMetaActions} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
            self.observed_sas = {state : {action : {next_state: False for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}

            # if i % (config.episodes // 100) == 0:
            print(f"Meta PRL-AGENT {self.model.planner}: episode {i}")

            done = False
            state = self.env.reset()
            planning = i < config.planning_steps

            while not done:
                # print(self.model.transition_function[state])
                if next_action is not None:
                    action = next_action
                    next_action = None
                    follows_meta_action = True

                elif planning:
                    if config.learn_meta_actions:
                        plan = self.model.plan(state, meta=True, observed_sas=self.observed_sas, meta_sas=self.meta_sas, meta_actions = [self.meta_actions_t, self.meta_actions_r])
                    else:
                        plan = self.model.plan(state, meta=True, observed_sas=self.observed_sas, meta_sas=self.meta_sas)
                    if isinstance(plan, tuple):
                        if len(plan) == 3:
                            action, target_action, target_state = plan
                            next_action = target_action
                            expected_next_state = target_state
                        elif len(plan) == 4:
                            action, target_action, target_state, target_reward = plan
                            next_action = target_action
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])
                
                if isinstance(action, BaseMetaActions):
                       print(state, action, target_action, target_state, target_reward)
                       if action == BaseMetaActions.INCREASE_REWARD:
                            prev_reward = self.model.reward_function[state, target_action, target_state]
                            self.model.update_reward(state, target_action, target_state, target_reward)
                       elif action == BaseMetaActions.INCREASE_TRANSITION_PROBABILITY:
                            self.model.update_transition_prob(state, target_action, target_state, 1.0)
                       self.meta_sas[state][target_action][target_state][action] = True
                       next_action = target_action
                elif isinstance(action, MetaAction):
                    if isinstance(action, MetaActionR):
                        prev_reward = self.model.reward_function[state, target_action, target_state]
                        self.model.update_reward(state, target_action, target_state, action.reward)

                    elif isinstance(action, MetaActionT):
                        self.model.update_transition_prob(state, target_action, target_state, 1.0)
                    self.meta_sas[state][target_action][target_state][action] = True
                    next_action = target_action
                else:
                    print(state, action)
                    next_state, reward, done, info = self.env.step(action)
                    if done and not info.get("TimeLimit.truncated"):
                        print("Completed ", i)
                    # print(state, action, next_state)
                    self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])

                    if planning:
                        if config.learn_meta_actions:
                            if not follows_meta_action:
                                if state != next_state and next_state != self.env.start:
                                    # print(state, next_state)
                                    action_sequence = self.model.action_sequence(state, next_state)
                                    if action_sequence != [action] and action_sequence is not None and len(action_sequence) <=3:
                                        meta_action = MetaActionT(action, action_sequence)
                                        if not meta_action in self.meta_actions_t:
                                            print(f"Learned: {meta_action}")
                                            self.meta_actions_t.append(meta_action)
                                if reward != self.model.reward_function[state, action, next_state]:
                                    meta_action = MetaActionR(reward)
                                    if not meta_action in self.meta_actions_r:
                                        print(f"Learned: {meta_action}")
                                        self.meta_actions_r.append(meta_action)
                            else:
                                follows_meta_action = False
                        self.N_sa[state][action]+=1
                        self.N_sas[state][action][next_state]+=1
                        self.model.update_transition_probs(state, action, self.N_sa[state][action], self.N_sas[state][action])
                        self.model.update_reward(state, action, next_state, reward)
                        self.observed_sas[state][action][next_state] = True

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
        with open("learnt_actions.txt", "w+") as fp:
            fp.write("\n".join([str(m) for m in list(set(self.meta_actions_r))]))
            fp.write("\n".join([str(m) for m in list(set(self.meta_actions_t))]))
        print(rewards)
        return rewards, states