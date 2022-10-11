import numpy as np
from copy import deepcopy as copy_deepcopy
from scipy import sparse
import sys
import matplotlib.pyplot as plt

import gym
from gym import spaces as gym_spaces

class ChainEnvironment(gym.Env):
    def __init__(self, env_args):
        super(ChainEnvironment, self).__init__()

        self.Rmax = env_args["Rmax"]
        self.terminal_state = 1
        self.actions = {"left": 0, "right": 1}
        self.actions_names = ["left", "right"]
        self.n_actions = len(self.actions)
        self.n_1 = env_args['n_1']
        self.n_2 = env_args['n_2']
        self.n_2_subgoal = env_args["n_2_subgoal"]
        self.n2_subgoal_position = env_args["n2_subgoal_position"]
        self.n_states = env_args["n_1"] + env_args["n_2"] + 2
        self.gamma = env_args["gamma"]
        self.randomMoveProb = env_args["randomMoveProb"]
        self.InitD = self.get_init_D()
        self.reward = self.get_reward()
        self.T = self.get_transition_matrix_line()
        self.T_sparse_list = self.get_transition_sparse_list()
        self.goal_states = self.get_goal_states()
        self.state_feature_matrix = self.get_state_feature_matrix()
        self.state_action_feature_matrix = self.get_state_action_feature_matrix()

        # # declare open gym necessary attributes
        self.observation_space = gym_spaces.Box(low=0.0, high=1.0, shape=(self.n_states,),
                                                dtype=np.float)
        self.action_space = gym_spaces.Discrete(self.n_actions)
        self.state = None
        self.H = env_args["H"]
        self.current_step = 1

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

    def get_init_D(self):
        InitD = np.zeros(self.n_states)/self.n_states
        InitD[self.n_2] = 1
        return InitD
    #enddef

    def get_reward(self):
        reward = np.zeros((self.n_states,self.n_actions))
        if self.terminal_state == 1:
            reward[-2, 1] = self.Rmax
        else:
            reward[-1, 1] = self.Rmax

        # # set n1 subgoal right action
        # reward[self.n_2 + 10, 1] = self.n_1_subgoal

        # set n2 subgoal left action
        reward[self.n_2 - self.n2_subgoal_position, 0] = self.n_2_subgoal
        return reward
    #enddef


    def get_state_feature_matrix(self):
        state_feature_matrix = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            state_feature_matrix[s, s] = 1
        return state_feature_matrix
    #enddef

    def get_state_action_feature_matrix(self):
        state_action_feature_matrix = np.zeros((self.n_states*self.n_actions,
                                         self.n_states*self.n_actions))
        for s_a in range(self.n_states*self.n_actions):
            state_action_feature_matrix[s_a, s_a] = 1
        return state_action_feature_matrix
    # enddef

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

    def get_goal_states(self):
        if self.terminal_state == 1:
            goal_states = [self.n_states-2]
        else:
            goal_states = [self.n_states-1]
        return goal_states
    #enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state
    #enddef

    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))  # T_0
        return T_sparse_list
    # endef

    def get_M_0(self):
        M_0 = (self.n_states, self.n_actions, self.reward, self.T, self.gamma, self.terminal_state)
        return M_0
    #enddef

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_clean_transition_matrix(self):
        P_clean = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.n_states):

            # left corner state
            if s == 0:
                # left action
                P_clean[s, -1, 0] = 1.0

                #right action
                P_clean[s, s+1, 1] = 1.0

            # right corner state
            elif s == self.n_states - 2:
                # right action
                P_clean[self.n_states - 2, self.n_states - 2, 1] = 1.0

                # left action
                P_clean[self.n_states - 2, self.n_states - 3, 0] = 1.0


            # middle states
            elif 0 < s < self.n_states - 2:
                # left action
                P_clean[s, s-1, 0] = 1.0

                # right action
                P_clean[s, s+1, 1] = 1.0

            # terminal state
            elif s == self.n_states - 1:
                P_clean[s, s, :] = 1.0

        return P_clean
    #enddef
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def get_transition_matrix_line(self):
        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        P_clean = self.get_clean_transition_matrix()

        for s in range(self.n_states):
            for a in range(self.n_actions):
                P[s, :, a] += (1-self.randomMoveProb) * P_clean[s, :, a]

                for a_prime in range(self.n_actions):
                    P[s, :, a_prime] += self.randomMoveProb * P_clean[s, :, a] / self.n_actions
        return P
    #enddef
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def get_transition_matrix_line_old(self):
        success_prob = 1 - self.randomMoveProb
        P = np.zeros((self.n_states, self.n_states, self.n_actions))

        for s in range(self.n_states):

            # left corner state
            if s == 0:
                # left action
                P[s, -1, 0] = success_prob
                P[s, s+1, 0] = self.randomMoveProb

                #right action
                P[s, -1, 1] = self.randomMoveProb
                P[s, s+1, 1] = success_prob

            # right corner state
            elif s == self.n_states - 2:
                # right action
                P[self.n_states - 2, self.n_states - 2, 1] = success_prob
                P[self.n_states - 2, self.n_states - 3, 1] = self.randomMoveProb

                # left action
                P[self.n_states - 2, self.n_states - 3, 0] = success_prob
                P[self.n_states - 2, self.n_states - 2, 0] = self.randomMoveProb


            # middle states
            elif 0 < s < self.n_states - 2:
                # left action
                P[s, s+1, 0] = self.randomMoveProb
                P[s, s-1, 0] = success_prob
                # right action
                P[s, s-1, 1] = self.randomMoveProb
                P[s, s+1, 1] = success_prob

            # terminal state
            elif s == self.n_states-1:
                P[s, s, :] = 1.0

        return P
    #enddef
#endclass

########################################
if __name__ == "__main__":



    env_args = {
        "Rmax": 1,
        "n_1": 10,
        "n_1_subgoal": 0.0,
        "n_2": 1,
        "n_2_subgoal": 0.0,
        "n_actions:": 2,
        "gamma": 0.999,
        "randomMoveProb": 0.0,
        "H": 50
    }
    env_orig = ChainEnvironment(env_args)

    print(np.sum(env_orig.T- env_orig.get_transition_matrix_line_old()))
    exit(0)

    for s in range(env_orig.n_states):

        print("T=", env_orig.T[s, :, :])
        print("T_old=", env_orig.get_transition_matrix_line_old()[s, :, :])

    exit(0)




    for s in range(env.n_states):
        for a in range(env.n_actions):
            print(sum(env.T[s,:, a]))
    print(env.n_states)
    print(env.InitD)
    input()
    exit(0)
    import MDPSolver

    Q, V, _, _ = MDPSolver.valueIteration(env, env.reward, tol=1e-10)
    print(V)
