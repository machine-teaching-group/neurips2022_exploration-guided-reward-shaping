import numpy as np
import copy
from scipy import sparse
import matplotlib.pyplot as plt
import math
import sys
import os

import gym
from gym import spaces as gym_spaces


class Environment(gym.Env):
    def __init__(self, env_args):
        super(Environment, self).__init__()

        # initialise MDP parameters
        self.actions = {"up": 0, "left": 1, "down": 2, "right": 3}
        self.actions_names = ["up", "left", "down", "right"]
        self.n_actions = len(self.actions)
        self.gridsizefull = env_args["gridsizefull"]
        self.R_max = env_args["R_max"]
        self.gamma = env_args["gamma"]
        self.terminal_state = 1
        self.randomMoveProb = env_args["randomMoveProb"]
        self.n2_subgoal = env_args["n2_subgoal"]
        self.wall_horizontal, self.wall_vertical, self.gates = self.compute_walls()
        self.gate_states = self.get_gate_states(self.gates)
        self.n_states = self.gridsizefull * self.gridsizefull if self.terminal_state == 0 \
            else self.gridsizefull * self.gridsizefull + 1
        self.InitD = self.get_InitD()
        self.reward = self.get_reward()
        self.T = self.get_transition_matrix()
        self.T_sparse_list = self.get_transition_sparse_list()
        self.goal_state = self.get_goal_states()

        # # declare open gym necessary attributes
        self.observation_space = gym_spaces.Box(low=0.0, high=1.0, shape=(self.n_states,),
                                                dtype=np.float)
        self.action_space = gym_spaces.Discrete(self.n_actions)
        self.state = None
        self.H = env_args["H"]
        self.current_step = 1

    #enddef

    def reset(self):
        self.current_step = 0
        state = np.random.choice(range(self.n_states), p=self.InitD)
        self.state = state
        return self.state
    #enddef

    def step(self, action:int):
        done = False
        self.current_step += 1
        # return: next_state, reward, done, info
        self.state, reward = self.sample_next_state_and_reward(self.state, action)

        if ((self.terminal_state == 1 and self.state == self.n_states-1)
                or self.H == self.current_step):
            done = True

        return self.state, reward, done, {}
    #enddef

    def sample_next_state_and_reward(self, s, a):
        reward = self.reward[s, a]
        next_s = self.sample_next_state(s, a)
        return next_s, reward
    #enddef

    def sample_next_state(self, s, a):
        next_s = np.random.choice(np.arange(0, self.n_states, dtype="int"),
                                      size=1, p=self.T[s, :, a])[0]
        return next_s
    #enddef

    def get_InitD(self):
        InitD = np.zeros(self.n_states)
        init_state = self.get_state_from_coord(1, 1)
        InitD[init_state] = 1
        return InitD
    #enddef

    def get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))
        if self.terminal_state == 0:
            reward[-1, 3] = self.R_max
        else:
            reward[-2, 3] = self.R_max

        # reward for n2 subgoal is given only for UP action
            n2_subgoal_x_coord = 1
            n2_subgoal_y_coord = 0
            state = self.get_state_from_coord(n2_subgoal_x_coord, n2_subgoal_y_coord)
            reward[state, 0] = self.n2_subgoal

        # reward
        return reward
    #enndef

    def get_goal_states(self):

        reward_soom_over_states = np.sum(self.reward, axis=1)
        goal_state = np.nonzero(reward_soom_over_states)[0]
        return goal_state
    #enddef

    def compute_wall_states(self, wall):
        wall_states = []
        for x, y in wall:
            wall_states.append(self.get_state_from_coord(x, y))
        return list(set(wall_states))
    #enddef

    def get_gate_states(self, gates):
        gate_states = []
        if self.gridsizefull%2==1:
            for x, y in gates:
                gate_states.append(self.get_state_from_coord(x, y))
            gate_states = sorted(gate_states)
            gate_states[0] -= 1
            gate_states[1] = gate_states[1]
            gate_states[2] -= self.gridsizefull
            gate_states[3] -= 1
        else:
            for x, y in gates:
                gate_states.append(self.get_state_from_coord(x, y))
            gate_states = sorted(gate_states)
            gate_states[0] = gate_states[0]
            gate_states[1] += self.gridsizefull
            gate_states[2] = gate_states[2]
            gate_states[3] = gate_states[3]

        return sorted(gate_states)
    #enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    def compute_walls(self):
        '''
        Returns:
            (list): Contains (x,y) pairs that define wall locations.
        '''
        walls_horizontal = []
        walls_vertical = []
        gates = []

        half_width = math.ceil(self.gridsizefull / 2.0)
        half_height = math.ceil(self.gridsizefull / 2.0)

        half_height_const = self.gridsizefull // 2 + 1
        half_width_const = self.gridsizefull//2

        # Wall from left to middle.
        for i in range(1, self.gridsizefull + 1):
            if i == half_width:
                half_height -= 1
            if i + 1 == math.ceil(self.gridsizefull / 3.0) or\
                    i == math.ceil(2 * (self.gridsizefull + 2) / 3.0):
                gates.append((i-1, half_height-1))
                continue

            walls_horizontal.append((i-1, half_height_const-1))

        # Wall from bottom to top.
        # half_width_const = copy.deepcopy(half_width)
        for j in range(1, self.gridsizefull + 1):
            if j + 1 == math.ceil(self.gridsizefull / 3.0) or \
                    j == math.ceil(2 * (self.gridsizefull + 2) / 3.0):
                gates.append((half_width-1, j-1))
                continue
            walls_vertical.append((half_width_const, j-1))


        return walls_horizontal, walls_vertical, gates
    #enddef

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_clean_transition_matrix(self):
        P_clean = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.gridsizefull*self.gridsizefull):
            possible_actions = self.get_possible_actions_within_grid(s)
            next_states = self.get_next_states(s, possible_actions)
            for a in range(self.n_actions):
                if a in possible_actions:
                    n_s = int(next_states[np.where(possible_actions == a)][0])  # direct next state
                    P_clean[s, n_s, a] = 1.0
                else:
                    P_clean[s, s, a] = 1.0


        if self.terminal_state == 1:
            #0th state
            P_clean[0, :, :] = 0
            #UP action
            P_clean[0, self.gridsizefull, 0] = 1

            #LEFT action
            P_clean[0, -1, 1] = 1

            #DOWN action
            P_clean[0, -1, 2] = 1

            #RIGHT action
            P_clean[0, 1, 3] = 1
            ################

            #goal state
            P_clean[-2, :, :] = 0

            #UP action
            P_clean[-2:, -2, 0] = 1

            # LEFT action
            P_clean[-2:, -3, 1] = 1

            #DOWN action
            P_clean[-2:, -self.gridsizefull-1, 2] = 1

            #RIGHT action
            P_clean[-2:, -2, 3] = 1

            #terminal self transition
            P_clean[-1, :, :] = 0
            P_clean[-1, -1, :] = 1

        return P_clean
    #enddef

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        P_clean = self.get_clean_transition_matrix()

        for s in range(self.n_states):
            for a in range(self.n_actions):
                P[s, :, a] += (1-self.randomMoveProb) * P_clean[s, :, a]

                for a_prime in range(self.n_actions):
                    P[s, :, a_prime] += self.randomMoveProb * P_clean[s, :, a] / self.n_actions
        return P
    #enddef


    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))
        return T_sparse_list
    # endef

    def state_to_wall_state(self, state):
        state_wall = np.array([a[0] + a[1] * self.gridsizefull for a in self.wall])
        # print(state_wall)
        n_less = len(np.where(state_wall <= state)[0])
        return state - n_less


    def get_possible_actions_within_grid(self, state):
        # Given a state, what are the possible actions from it
        possible_actions = []
        state_x, state_y = state % self.gridsizefull, state // self.gridsizefull
        # print(state_x, state_y)
        # print((state_x-1, state_y) in self.wall)
        if ((state_y < self.gridsizefull-1) and (state_x, state_y+1) not in self.wall_horizontal): #UP action is not allowed from down state of the horizontal wall
            possible_actions.append(self.actions["up"])
        if ((state_y > 0) and (state_x, state_y) not in self.wall_horizontal): #DOWN action is not allowed from horizontal wall state
            possible_actions.append(self.actions["down"])
        if ((state_x > 0) and (state_x, state_y) not in self.wall_vertical): #LEFT action is not allowed from the vertical wall_state
            possible_actions.append(self.actions["left"])
        if ((state_x < self.gridsizefull - 1) and (state_x+1, state_y) not in self.wall_vertical): #RIGHT action is not allwed from the left side of the vertical wall
            possible_actions.append(self.actions["right"])
        # possible_actions.append(self.actions["stay"])
        possible_actions = np.array(possible_actions, dtype=np.int)
        return possible_actions
    #enddef

    def get_next_states(self, state, possible_actions):
        # Given a state, what are the posible next states I can reach
        next_state = []
        state_x, state_y = state % self.gridsizefull, state // self.gridsizefull
        for a in possible_actions:
            if a == 0: next_state.append((state_y+1) * self.gridsizefull + state_x)
            elif a == 1: next_state.append(state_y * self.gridsizefull + state_x-1)
            elif a == 2: next_state.append((state_y-1) * self.gridsizefull + state_x)
            elif a == 3: next_state.append(state_y * self.gridsizefull + state_x + 1)
            # else: next_state.append(state)
        next_state = np.array(next_state, dtype=np.int)
        return next_state

    def is_wall(self, x, y):
        state_coord = (x, y)
        if state_coord in self.wall_horizontal or state_coord in self.wall_vertical:
            return True
        return False
    #enddef

    def is_wall(self, s):
        state_x, state_y = s % self.gridsizefull, s // self.gridsizefull
        state_coord = (state_x, state_y)
        if state_coord in self.wall_horizontal or state_coord in self.wall_vertical:
            return True
        return False
    #enddef

    def get_state_from_coord(self, x, y):
        # state_wall = np.array([a[0] + a[1]*self.gridsizefull for a in self.wall])
        # print(state_wall)
        orig_state = y * self.gridsizefull + x
        # n_less = len(np.where(state_wall<=orig_state)[0])
        return orig_state
    #enddef

    def convert_det_to_stochastic_policy(self, deterministicPolicy):
        # Given a deterministic Policy, I will return a stochastic policy
        n_states = self.gridsizefull*self.gridsizefull
        stochasticPolicy = np.zeros((n_states, self.n_actions))
        for i in range(n_states):
            stochasticPolicy[i][deterministicPolicy[i]] = 1
        return stochasticPolicy

    # enddef

    def draw(self, V, pi, reward, show, strname, fignum):
        f = fignum
        n_states = self.n_states
        # plt.figure(f)

        # pi = copy.deepcopy(pi)
        # pi = pi.reshape(self.gridsizefull, self.gridsizefull)
        # pi = pi.flatten()


        if len(pi.shape) == 1:
            pi = self.convert_det_to_stochastic_policy(pi)
        # plt.pcolor(reward)
        # plt.title(strname + "Reward")
        # plt.colorbar()

        # print(pi.shape)

        if self.terminal_state == 1:
            V = copy.deepcopy(V[:-1])
            pi = np.delete(pi, (-1), axis=0)
            n_states = self.n_states-1

        f += 1
        if V is not None:
            plt.figure(f)
            reshaped_Value = copy.deepcopy(V.reshape((self.gridsizefull, self.gridsizefull)))
            plt.pcolor(reshaped_Value)
            plt.colorbar()
            x = np.linspace(0, self.gridsizefull - 1, self.gridsizefull) + 0.5
            y = np.linspace(0, self.gridsizefull - 1, self.gridsizefull) + 0.5
            X, Y = np.meshgrid(x, y)
            zeros = np.zeros((self.gridsizefull, self.gridsizefull))
            if pi is not None:
                for a in range(self.n_actions):
                    pi_ = np.zeros(n_states)
                    for s in range(n_states):
                        pi_[s] = 0.45*pi[s, a]/np.max(pi[s, :])

                    pi_ = (pi_.reshape(self.gridsizefull, self.gridsizefull))
                    if a == 0:
                        plt.quiver(X, Y, zeros, pi_, scale=1, units='xy')
                    elif a == 1:
                        plt.quiver(X, Y, -pi_, zeros, scale=1, units='xy')
                    elif a == 2:
                        plt.quiver(X, Y, zeros, -pi_, scale=1, units='xy')
                    elif a == 3:
                        plt.quiver(X, Y, pi_, zeros, scale=1, units='xy')
            plt.title(strname + "Opt values and policy")
        if(show == True):
            #print(" show is true")
            plt.show()
    #enddef

#endclass

if __name__ == "__main__":

    # gridsizefull_array = [7]
    # H_array = [4]
    #
    # acc_dict = {}

    env_args = {
        "gridsizefull": 11,
        "R_max": 10,
        "gamma": 0.99,
        "randomMoveProb": 0.0,
        "n1_subgoal": 0.0,
        "n2_subgoal": 0.0,
        "H": 50
    }

    env = Environment(env_args)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            print(sum(env.T[s,:, a]))
    print()
    # print(np.sum(env.T- env.get_transition_matrix_old(randomMoveProb=0.3)))

    print(env.reward)

    import MDPSolver

    Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward)
    # print(env.wall_vertical)
    # print(env.wall_horizontal)
    print(env.gate_states)
    print(env.get_state_from_coord(0, 4))
    env.draw(V, pi_s, env.reward, True, strname="some", fignum=3)
    exit(0)
