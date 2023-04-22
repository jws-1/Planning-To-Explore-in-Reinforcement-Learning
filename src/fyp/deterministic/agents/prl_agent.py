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
        self.Q = {state : {action : 100. for action in range(self.env.nA)} for state in range(self.env.nS)}
        self.model = deepcopy(self.initial_model)

    def learn(self, config):
        self.reset()

        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, self.env.nS))

        for i in range(config.episodes):
            # if i % (config.episodes // 100) == 0:
            print(f"PRL-AGENT: episode {i}")

            done = False
            state = self.env.reset()

            planning = i < config.planning_steps

            while not done:
            

                if planning:
                    if random.uniform(0, 1) < config.eps:
                        action = self.env.action_space.sample()
                    else:
                        action = self.model.plan_VI(state)
                else:
                    action = random.choice([a for a in range(self.env.nA) if self.Q[state][a] == max(self.Q[state].values())])
                
                next_state, reward, done, _ = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + config.lr * ((reward + max(self.Q[next_state].values())) - self.Q[state][action])

                if config.learn_model and planning:
                    self.model.update_transition(state, action, next_state)
                    self.model.update_reward(state, action, reward)

                if state != next_state:
                    states[i][state]+=1
                
                state = next_state
                rewards[i] += reward


            states[i][state]+=1
        return rewards, states
