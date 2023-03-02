from rl_agent import RLAgent
from copy import deepcopy
import numpy as np
from meta_planner import MetaPlanner
from actions import Action, MetaAction, ACTION_MODIFIERS
from gridworld import GridWorld
import operator
from types import SimpleNamespace
from sys import exit
from watchpoints import watch
class RLMetaAgent(RLAgent):

    def __init__(self, world, model):
        self.world = world
        self.initial_model = deepcopy(model)
        self.model = model
        self.q_table = np.full((*self.world.dim, 4), 0., dtype=float)

    def reset(self):
        self.q_table = np.full((*self.world.dim, 4), 0., dtype=float)
        self.model = deepcopy(self.initial_model)
        self.planner = MetaPlanner(self.model)
    
    def learn(self, config):
        self.reset()
        rewards = np.zeros(config.episodes)
        states = np.zeros((config.episodes, *self.world.dim))
        observed_obstacles = np.zeros((config.episodes, *self.world.dim))
        meta_calls = np.zeros((config.episodes, *self.world.dim, 4))
        meta_calls[:, self.world.start[0], self.world.start[1], :] = 1.
        actions = [[] for _ in range(config.episodes)]
        steps = [[] for _ in range(config.episodes)]



        for i in range(config.episodes):

            if i % 100 == 0:
                print(f"Meta Agent, episode {i}")

            planning = i < config.planning_steps
            episodic_observations = np.full((self.world.dim), False, dtype=bool)
            episodic_observations[self.world.start] = True
            if i < config.static_threshold:
                self.world.sample()
            else:
                print("static")
                self.world.static()
            
            done = False
            state = self.world.current
            # states[i][state]+=1
            meta_calls[i, :, :, :] = 0.
            observed_obstacles[i, :, :] = 0.
            while not done:
                self.model.current = deepcopy(self.world.current)
                if planning:
                    if i == 0:
                        P = observed_obstacles[i]
                    else:
                        P = np.mean(observed_obstacles[max(0, i-config.memory_episodes):i], axis=0)
                        #P = np.sum(observed_obstacles[:i % config.memory_episodes], axis=0) / max(observed_obstacles[:i % config.memory_episodes].shape[0], 1) #observed_obstacles[max(i-config.memory_episodes, 0):i].shape[0]
                    self.model.obstacles = {}
                    for k in range(P.shape[0]):
                        for j in range(P.shape[1]):
                            if P[k,j] >= 0.5:
                                self.model.obstacles[(k,j)] = P[k,j]
                            if observed_obstacles[i][k][j] > 0.0:
                                self.model.obstacles[(k,j)] = 1.0
                    # beliefs[i] = P[(3,7)]
                    # actuals[i] = 1 if (3,7) in self.world.episodic_obstacles else 0
                if planning:
                    if i == 0:
                        episodic_meta_calls = meta_calls[i]
                    else:
                        episodic_meta_calls = np.sum(meta_calls[max(0, i-config.memory_episodes):(i+1)], axis=0)
                        # episodic_meta_calls = np.sum(meta_calls[:max((i+1) % config.memory_episodes, meta_calls.shape[0])], axis=0)
                    action, posterior = self.planner.plan(episodic_meta_calls, episodic_observations)
                else:
                    action, posterior = Action(int(np.argmax(self.q_table[state]))), None

                if isinstance(action, Action):
                    reward, done = self.world.action(action, posterior)
                else:
                    reward, done = self.model.action(action, posterior)
                actions[i].append(action)
                steps[i].append(self.world.current)
                next_state = self.world.current

                if isinstance(action, Action):
                    old_value = self.q_table[state[0]][state[1]][action.value]
                    next_max = np.max(self.q_table[next_state])
                    new_value = (1 - config.lr) * old_value + config.lr * (reward + config.df * next_max)
                    self.q_table[state[0]][state[1]][action.value] = new_value
                    if planning:
                        episodic_observations[posterior] = True
                        expected_state = tuple(map(operator.add, state, ACTION_MODIFIERS[action]))
                        expected_reward = self.model.rewards[expected_state]
                        if self.model.ok_state(expected_state, episodic=False):
                            if self.world.current != expected_state:
                                observed_obstacles[i][expected_state[0]][expected_state[1]] = 1.0
                            elif expected_reward != reward:
                                self.model.rewards[expected_state] = reward
                else:
                    meta_calls[i][posterior[0]][posterior[1]][action.value] = 1.0
                
                states[i][state]+=1
                state = next_state
                
                rewards[i]+=reward
            states[i][state]+=1

        return rewards, states

"""
Agent develops a belief about the state space (transition probabilities < 0.5 are considered obstacles), based on the last k episodes.
Observations are episodic; observations from the current episode cannot be contradicted.
Meta calls expire after k observations, this is the memory length.
"""

if __name__ == "__main__":
    dim = (10,10)
    start = (0,0)
    goal = (9,9)

    world_rewards = np.full(dim, -2, dtype=int)

    world_rewards[1][0] = -1
    world_rewards[2][0] = -1
    world_rewards[3][0] = -1
    
    world_obstacles = {
        (0,3) : 0.5,
        (1,3) : 0.1,
        (2,3) : 0.5,
        (3,3) : 1.0,
        (4,3) : 1.0,
        (5,3) : 0.001,
        (6,3) : 0.5,
        (7,3) : 0.3,
        (8,3) : 0.5,
        (9,3) : 1.0
    }

    static_obstacles = [
        (0,3), (1,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3)
    ]


    world = GridWorld(dim, start, goal, world_obstacles, world_rewards, static_obstacles=static_obstacles, static_rewards=world_rewards)

    model = GridWorld(dim, start, goal, {},np.full(dim, -2, dtype=int))

    agent = RLMetaAgent(world, model)

    config = {
        "episodes": 1000,
        "m":20,
        "lr":0.6,
        "df":1.0, # episodic, so rewards are undiscounted.
        "window_size":20,
        "planning_steps" : 50,
        "learn_model" : True,
        "memory_episodes": 5,
        "static_threshold": 250
    }
    rewards, rewards_95pc, states = agent.learn_and_aggregate(SimpleNamespace(**config))

    min_, max_ = 0, 1000
    print(states.shape)
    agent.plot_results(rewards[min_:max_], states[:, min_:max_, :, :], rewards_95pc=rewards_95pc[min_:max_,:], policy="meta", save=True)
    print(f"Meta low, mean, high, final planning, final model-free rewards: {np.min(rewards), np.mean(rewards), np.max(rewards), rewards[config['planning_steps']-config['window_size']], rewards[-1]}")