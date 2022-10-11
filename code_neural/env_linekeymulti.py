import numpy as np
import copy
import itertools
import random

import gym
from gym import spaces as gym_spaces


class Environment(gym.Env):
    def __init__(self, env_args):
        super(Environment, self).__init__()
        self.randomMoveProb = env_args["randomMoveProb"]
        self.R_max = env_args["R_max"]
        self.n_picks = env_args["n_picks"]
        self.n_actions = env_args["n_actions"]
        # self.actions = np.array([0, 1, 2])
        self.gamma = env_args["gamma"]
        self.H = env_args["H"]
        self.terminal_state = 1
        self.delta_move = [0.074, 0.076]
        # self.delta_move = [0.049, 0.051]
        self.reward_range = [0.9, 1]
        self.key_location = [0.0, 0.1]
        self.small_noise = [0, 0]
        self.current_state = None # [x, flag_agent_has_key, flag_x_is_key_area, flag_x_is_goal_area]
        self.start_state = env_args["init_state"]
        self.finish_action_termination_flag = env_args["finish_action_termination_flag"]
        self.small_reward_for_picking_key = env_args["small_reward_for_picking_key"]
        self.small_reward_for_goal_without_key = env_args["small_reward_for_goal_without_key"]
        self.random_small_reward_for_goal_without_key = env_args["random_small_reward_for_goal_without_key"]
        # self.negative_reward_for_pick_action = 0.0
        self.steps = None
        self.done = False
        self.reward = None

        # # declare open gym necessary attributes
        self.observation_space = gym_spaces.Box(low=0.0, high=1.0, shape=(4 + self.n_picks,),
                                                dtype=np.float)
        self.action_space = gym_spaces.Discrete(self.n_actions)
    #enddef

    def reset(self):
        flag_agent_has_key = 0
        flag_x_is_key_area = 0
        flag_x_is_goal_area = 0
        flag_array_for_different_keys = [0] * self.n_picks


        x = random.uniform(self.start_state[0],
                           self.start_state[1])

        # x = (self.start_state[0] + self.start_state[1]) / 2

        if self.key_location[0] <= x <= self.key_location[1]:
            flag_x_is_key_area = 1.0

        if self.reward_range[0] <= x <= self.reward_range[1]:
            flag_x_is_goal_area = 1.0

        self.current_state = ([x, flag_agent_has_key, flag_x_is_key_area, flag_x_is_goal_area] + flag_array_for_different_keys)
        self.steps = 1
        self.done = False
        return self.current_state
    #enddef

    def step(self, action):

        next_state = self.get_transition(self.current_state, action)
        reward = self.get_reward(self.current_state, action, next_state)

        # if reward > 0:
        #     print(self.current_state, reward, next_state, action)
        #     input()

        if self.steps > self.H or next_state[0] == -1:
            self.done = True

        self.current_state = next_state
        self.steps += 1
        return next_state, reward, self.done, {}
    #enddef

    def get_transition(self, state,  action):

        if action >= 2: # pick action has no random move
            return self.get_clean_transition(state, action)

        # if action == 3: # finish action has no random move
        #     return self.get_clean_transition(action)

        random_number = np.random.random()

        if random_number >= self.randomMoveProb:
            return self.get_clean_transition(state, action)
        else:
            return self.get_clean_transition(state, 1 - action) # take opposite action

    #enddef

    def get_clean_transition(self, state, action):
        next_state = None
        if action == 0:  # LEFT action
            next_state = self.transitions_LEFT(state)

        elif action == 1:  # RIGHT action
            next_state = self.transitions_RIGHT(state)

        elif action >= 2:  # PICK action
            next_state = self.transitions_pick_KEY(state, action)

        # elif action == 3:  # FINISH action
        #     next_state = self.transitions_FINISH()

        else:
            print(action)
            exit(0)

        return next_state

    def transitions_LEFT(self, current_state):
        next_state = None
        flag_agent_has_key = current_state[1]
        flag_x_is_key_area = 0.0
        flag_x_is_goal_area = 0.0
        flag_array_for_different_keys = current_state[4:]

        move_size_x = - random.uniform(self.delta_move[0], self.delta_move[1]) +\
                      random.uniform(self.small_noise[0], self.small_noise[1])

        next_x = current_state[0] + move_size_x

        if next_x < 0.0:
            next_x = 0.0

        if self.key_location[0] <= next_x <= self.key_location[1]:
            flag_x_is_key_area = 1.0

        if self.reward_range[0] <= next_x <= self.reward_range[1]:
            flag_x_is_goal_area = 1.0

        next_state = [next_x, flag_agent_has_key, flag_x_is_key_area, flag_x_is_goal_area] + flag_array_for_different_keys

        return next_state
    #enddef

    def transitions_RIGHT(self, current_state):
        next_state = None
        flag_agent_has_key = current_state[1]
        flag_x_is_key_area = 0.0
        flag_x_is_goal_area = 0.0
        flag_array_for_different_keys = current_state[4:]

        move_size_x = random.uniform(self.delta_move[0], self.delta_move[1]) +\
                      random.uniform(self.small_noise[0], self.small_noise[1])

        next_x = current_state[0] + move_size_x

        if next_x > 1.0:
            next_x = 1.0

        if self.key_location[0] <= next_x <= self.key_location[1]:
            flag_x_is_key_area = 1.0

        if self.reward_range[0] <= next_x <= self.reward_range[1]:
            flag_x_is_goal_area = 1.0

        next_state = [next_x, flag_agent_has_key, flag_x_is_key_area, flag_x_is_goal_area] + flag_array_for_different_keys

        return next_state
    #enddef

    def transitions_pick_KEY(self, current_state, action):
        next_state = None
        next_x = current_state[0]
        flag_agent_has_key = current_state[1]
        flag_x_is_key_area = current_state[2]
        flag_x_is_goal_area = current_state[3]
        flag_array_for_different_keys = current_state[4:]

        if(flag_agent_has_key == 0) and \
                (self.key_location[0] <= current_state[0] <= self.key_location[1]):

            if (action == 2):
                flag_array_for_different_keys[action - 2] = 1.0
                flag_agent_has_key = 1.0
                next_state = [next_x, flag_agent_has_key, flag_x_is_key_area,
                              flag_x_is_goal_area] + flag_array_for_different_keys
            else:
                random_index = np.random.choice(range(1, self.n_picks))
                flag_array_for_different_keys[random_index] = 1.0
                flag_agent_has_key = 1.0
                next_state = [next_x, flag_agent_has_key, flag_x_is_key_area,
                              flag_x_is_goal_area] + flag_array_for_different_keys
        else:
            next_state = [next_x, flag_agent_has_key, flag_x_is_key_area,
                          flag_x_is_goal_area] + flag_array_for_different_keys

        return next_state
    #enddef

    def get_reward(self, state, action, next_state=None):

        # return reward if agent took RIGHT action from goal and it has 1st key
        if (action == 1) and (self.reward_range[0] <= state[0] <= self.reward_range[1]) \
                and (state[4] == 1):
            return self.R_max

        # return small if agent took RIGHT action from goal with NO 1st key
        if (action == 1) and (self.reward_range[0] <= state[0] <= self.reward_range[1]) \
                and (state[4] == 0):
            if random.random() > 0.5:
                return self.small_reward_for_goal_without_key + self.random_small_reward_for_goal_without_key
            else:
                return self.small_reward_for_goal_without_key - self.random_small_reward_for_goal_without_key

        # return small if agent took 1st key
        if (action == 2) and (self.key_location[0] <= state[0] <= self.key_location[1])\
                and (state[1] == 0):
            return self.small_reward_for_picking_key

        return 0
    #enddef

#endclass

if __name__ == "__main__":

    n_picks = 3

    env_args = {
        "R_max": 1,
        "gamma": 0.99,
        "randomMoveProb": 0.05,
        "n_picks": n_picks,
        "n_actions": 2 + n_picks,
        "H": 30,
        "init_state": [0.0, 0.1],
        "finish_action_termination_flag": 0.0,
        "small_reward_for_picking_key": 0.0,
        "small_reward_for_goal_without_key": 0.0
    }
    env = Environment(env_args)


    state = env.reset()
    while True:

        action = np.random.choice(range(0, n_picks+2))
        next_state, reward, done, _ = env.step(action)

        print('s={}, a={}, r={}, next_s={}'.format(state, action, reward, next_state))

        state = next_state

        if done:
            break