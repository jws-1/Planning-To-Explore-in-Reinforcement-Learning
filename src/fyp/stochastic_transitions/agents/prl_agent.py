from copy import deepcopy
from .rl_agent import RLAgent
import numpy as np
import random

class PRLAgent(RLAgent):

    def __init__(self, env, model):
        self.env = env
        self.initial_model = model
        self.reset()

    def reset(self):
        self.Q = {state : {action : 0. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)
        self.N_sa = {state : {action : 0 for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.N_sas = {state : {action : {next_state : 0 for next_state in range(self.env.nS)} for action in range(self.env.nA)} for state in range(self.env.nS)}

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))

        # if config.decay_lr:
        #     decay_factor = (config.min_lr/config.lr)**(1/config.episodes)
        # else:
        #     decay_factor = 1.0
        decay_factor = 0.0001
        lr = config.lr

        for i in range(config.episodes):
            # if i % (config.episodes // 100) == 0:
            print(f"PRL-AGENT: episode {i}")

            done = False
            state = self.env.reset()

            planning = i < config.planning_steps

            while not done:
                # self.env.render()

                if planning:
                    # if random.uniform(0, 1) < config.eps:
                        # action = self.env.action_space.sample()
                    # else:
                    action = self.model.plan_VI(state)
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])
                next_state, reward, done, info = self.env.step(action)
                # print(state, action, next_state)
                if done and not info.get("TimeLimit.truncated"):
                    print("Completed ", i)
                # old_value = self.Q[state][action]
                # next_max = max(self.Q[next_state].values())
                # new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                self.Q[state][action] = self.Q[state][action] + lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])

                if config.learn_model and planning:
                    self.N_sa[state][action]+=1
                    self.N_sas[state][action][next_state]+=1
                    self.model.update_transition_probs(state, action, self.N_sa[state][action], self.N_sas[state][action])
                    self.model.update_reward(state, action, next_state, reward)

                if state != next_state:
                    states[i][state]+=1
                
                state = next_state
                rewards[i] += reward

            states[i][state]+=1
            # if not planning:
            #     lr = config.lr * (decay_factor ** i)
        return rewards, states
