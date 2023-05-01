from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random
from ..actions import BaseMetaActions, MetaActionR, MetaActionT, MetaAction
from collections import defaultdict
class MetaPRLAgent(RLAgent):

    def __init__(self, env, model):
        self.env = env
        self.initial_model = model
        self.reset()

    def reset(self):
        self.meta_actions = []
        self.Q = {state : {action : 0. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)
        self.observed_sa = {state : {action : False for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.meta_sa = {state : {action : {meta_action : False for meta_action in BaseMetaActions} for action in range(self.env.nA)} for state in range(self.env.nS)}
        # self.meta_sas = {state : {action : {next_state : {meta_action: False for meta_action in MetaAction} for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.meta_actions_r = []
        self.meta_actions_t = []


    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))
        next_action = None
        for i in range(config.episodes):
            # if i % (config.episodes // 10) == 0:
            print(f"Meta PRL-AGENT {self.model.planner}: episode {i}")

            done = False
            state = self.env.reset()

            planning = i < config.planning_steps
            follows_meta_action = False
            while not done:
                
                if next_action is not None:
                    action = next_action
                    next_action = None
                    follows_meta_action = True

                elif planning:
                    if not config.learn_meta_actions:
                        plan = self.model.plan(state, meta=True, observed_sa=self.observed_sa, meta_sa=self.meta_sa)
                    else:
                        plan = self.model.plan(state, meta=True, observed_sa=self.observed_sa, meta_sa=self.meta_sa, meta_actions=[self.meta_actions_t, self.meta_actions_r,], use_learned_actions=True)
                    if isinstance(plan, tuple):
                        if isinstance(plan[0], BaseMetaActions):
                            if plan[0] == BaseMetaActions.ADD_TRANSITION:
                                action, target_action, target_state = plan
                            elif plan[0] == BaseMetaActions.INCREASE_REWARD:
                                action, target_action, target_reward = plan
                        elif isinstance(plan[0], MetaActionR):
                            action, target_action, target_reward = plan
                        elif isinstance(plan[0], MetaActionT):
                            action, target_action, target_state = plan
                    else:
                        action = plan
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])
                if isinstance(action, BaseMetaActions):
                    if action == BaseMetaActions.ADD_TRANSITION:
                        print(state, action, target_action, target_state)
                        self.model.update_transition(state, target_action, target_state)
                    elif action == BaseMetaActions.INCREASE_REWARD:
                        print(state, action, target_action, target_reward)
                        self.model.update_reward(state, target_action, target_reward)
                    self.meta_sa[state][target_action][action] = True
                elif isinstance(action, MetaAction):
                    if isinstance(action, MetaActionR):
                        print(state, action, target_action)
                        self.model.update_reward(state, target_action, target_state, target_reward)
                    elif isinstance(action, MetaActionT):
                        print(state, action, target_action, target_state)
                        self.model.update_transition(state, target_action, target_state)
                    self.meta_sa[state][target_action][action] = True
                    print(self.meta_sa[state][target_action], self.model.planner)
                else:
                    print(state, action)
                    next_state, reward, done, _ = self.env.step(action)
                    self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])
                    if planning:

                        if config.learn_meta_actions:
                            if not follows_meta_action:
                                if state != next_state:
                                    action_sequence = self.model.action_sequence(state, next_state)
                                    if action_sequence != [action] and action_sequence is not None:
                                        meta_action = MetaActionT(action, action_sequence)
                                        if not meta_action in self.meta_actions_t:
                                            print(f"Learned: {meta_action}")
                                            self.meta_actions_t.append(meta_action)
                                            for s_ in range(self.env.nS):
                                                for a_ in range(self.env.nA):
                                                    self.meta_sa[s_][a_][meta_action] = False
                                if reward != self.model.reward_function[state, action]:
                                    meta_action = MetaActionR(reward)
                                    if not meta_action in self.meta_actions_r:
                                        print(f"Learned: {meta_action}")
                                        self.meta_actions_r.append(meta_action)
                                        for s_ in range(self.env.nS):
                                            for a_ in range(self.env.nA):
                                                self.meta_sa[s_][a_][meta_action] = False
                            else:
                                follows_meta_action = False

                        self.model.update_transition(state, action, next_state)
                        self.model.update_reward(state, action, reward)
                        self.observed_sa[state][action] = True


                    states[i][state]+=1
                
                    state = next_state
                    rewards[i] += reward    

            states[i][state]+=1
        with open("learnt_actions.txt", "w+") as fp:
            fp.write("\n".join([str(m) for m in self.meta_actions]))
        return rewards, states