import numpy as np
import torch
import gym
import copy
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvTeacher(gym.Env):

    def __init__(self, env, args, teacher_name):
        super(EnvTeacher, self).__init__()
        # self.teachers = ["orig", "ExploB", "SelfRS", "ExploRS", "sors", "lirpg"]
        self.teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "SORS_with_Rbar", "LIRPG_without_metagrad"]

        if teacher_name not in self.teachers:
            print("Error!!!")
            print(teacher_name)
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        self.env = env
        self.args = args
        self.teacher_name = teacher_name

        # declare open gym necessary attributes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma
        self.ExploB_w = np.zeros(self.n_states)
        self.phi_SelfRS = np.zeros((self.n_states, self.n_actions))
        self.phi_sors = np.zeros((self.n_states, self.n_actions))
        self.phi_lirpg = np.zeros((self.n_states, self.n_actions))
        self.V = np.zeros(self.n_states)
        self.curr_state = None
        self.first_succ_episode_number = None
        self.count_episode = 0.0
        self.goal_visits = 0.0
        self.episode_goal_visited = None

    # enddef

    def step(self, action):
        self.curr_state = self.env.state
        if (self.curr_state == self.n_states - 2) and \
            (not self.episode_goal_visited):
            self.episode_goal_visited = True
            self.goal_visits += 1.0

        next_state, reward_orig, done, info = self.env.step(action)

        if self.teacher_name in ["Orig", "SelfRS"]:
            r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action)

        elif self.teacher_name in ["ExploB", "ExploRS"]:
            r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action, next_state)

            #update R_explore
            self.update_ExploB_given_state(self.curr_state)

        elif self.teacher_name in ["SORS_with_Rbar"]:
            r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action)

        elif self.teacher_name in ["LIRPG_without_metagrad"]:
            r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action)

        else:
            print("Error in TeacherEnv.step()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)

        return next_state, r_hat, done, info
    # enddef

    def reset(self):
        self.episode_goal_visited = False
        return self.env.reset()
    # enddef

    def get_r_addition_term(self, state, action, next_state=None):
        if self.teacher_name == "Orig":
            return 0

        elif self.teacher_name == "ExploB":
            return self.R_explore(next_state)
            # return self.R_explore(state)

        elif self.teacher_name == "SelfRS":
            return self.R_exploit(state, action)

        elif self.teacher_name == "ExploRS":
            return self.R_ExploRS(state, action, next_state)

        elif self.teacher_name == "SORS_with_Rbar":
            return self.R_sors(state, action)

        elif self.teacher_name == "LIRPG_without_metagrad":
            return self.R_lirpg(state, action)


        else:
            print("Error in TeacherEnv.get_r_addition_term()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
    # enddef

    def R_explore(self, state):

        if self.env.terminal_state == 1 and \
                state == self.env.n_states - 1:
            return 0.0

        numerator = self.args["ExploB_lmbd"]
        N_s = np.power(self.args["ExploB_lmbd"] / self.args["ExploB_max"], 2.0) + self.ExploB_w[state]
        denominator = np.sqrt(N_s)
        # print(numerator/denominator)
        return numerator/denominator
    # enddef

    def R_exploit(self, state, action):
        return self.phi_SelfRS[state, action]
    # enddef

    def R_ExploRS(self, state, action, next_state=None):
        return self.R_exploit(state, action) + self.R_explore(next_state)
    #enddef

    def R_sors(self, state, action):
        return self.phi_sors[state, action]
    #enddef

    def R_lirpg(self, state, action):
        return self.phi_lirpg[state, action]
    #enddef

    def update(self, D, agent=None):

        if self.teacher_name == "Orig":
            pass

        elif self.teacher_name == "ExploB":
            pass

        elif self.teacher_name == "SelfRS":
            return self.update_SelfRS(D)

        elif self.teacher_name == "ExploRS":
            return self.update_ExploRS(D)

        elif self.teacher_name == "SORS_with_Rbar":
            self.update_sors(D)

        elif self.teacher_name == "LIRPG_without_metagrad":
            self.update_Rlirpg(D, agent)


        else:
            print("Error in TeacherEnv.update()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
    #enddef

    def update_ExploB_given_state(self, state):
        self.ExploB_w[state] += 1.0
    #enddef

    def update_ExploB_given_batch(self, D):

        for episode in D:
            for state, _, _, _, _, _, in episode:
                self.ExploB_w[state] += 1.0
    # enddef

    def update_Rlirpg(self, D, agent):
        postprocessed_D = self.postprocess_data(D)

        phi_rlirpg_grad_accumulator = np.zeros((self.n_states, self.n_actions))

        for episode in postprocessed_D:
            phi_SelfRS_grad_accumulator_Q = np.zeros((self.n_states, self.n_actions))
            phi_SelfRS_grad_accumulator_V = np.zeros((self.n_states, self.n_actions))
            for state_i, action_i, _, _, _, pi_given_s, _, G_bar in episode:

                ## calculate trajectory from every state_i, action_i on original env
                trajectory_s_i_a_i = utils.sample_trajectory_given_state_action(self.env,
                                                                                agent, start_state=state_i,
                                                                                start_action=action_i, epsilon_reinforce=0.0,
                                                                                H=self.env.H)
                # postprocessed_trajectory_s_i_a_i = self.get_postposessed_episode(self.env, trajectory_s_i_a_i)

                sampled_action_b = np.random.choice(range(len(pi_given_s)), p=pi_given_s)
                # sampled_action_b = np.random.choice(range(len(pi_given_s)))
                trajectory_s_i_b = utils.sample_trajectory_given_state_action(self.env,
                                                                                agent, start_state=state_i,
                                                                                start_action=sampled_action_b, epsilon_reinforce=0.0,
                                                                              H=self.env.H)


                #Compute gradient of Q(s_i, a_i)
                for t, (s_i, a_i, _, _, _, _) in enumerate(trajectory_s_i_a_i):
                    phi_SelfRS_grad_accumulator_Q[s_i, a_i] += (self.gamma**t)

                #Compute gradient of randomly sampled action_b
                for t, (s_i, b_i, _, _, _, _) in enumerate(trajectory_s_i_b):
                    phi_SelfRS_grad_accumulator_V[s_i, b_i] += (self.gamma**t)

                final_gradient = phi_SelfRS_grad_accumulator_Q - phi_SelfRS_grad_accumulator_V

                # compute gradient
                phi_rlirpg_grad_accumulator[state_i, action_i] += (pi_given_s[action_i] *
                                                                   (G_bar - self.V[state_i])) * \
                                                                  final_gradient[state_i, action_i]
        # Update phi_ExploB
        self.phi_lirpg += self.args["eta_phi_rlirpg"] * phi_rlirpg_grad_accumulator / len(D)

        # declare gradient for critic to do average update
        critic_grad = np.zeros(self.n_states)
        total_critic = 0.0
        # V update
        for episode in postprocessed_D:
            for state, _, _, _, _, _, _, G_bar in episode[::-1]:
                delta = (G_bar - self.V[state])
                critic_grad[state] += delta
                total_critic += 1.0

        # Update gradient critic
        self.V += self.args["eta_critic"] * critic_grad / total_critic

    def update_SelfRS(self, D):

        postprocessed_D = self.postprocess_data(D)

        phi_SelfRS_grad_accumulator = np.zeros((self.n_states, self.n_actions))

        for episode in postprocessed_D:
            for state_i, action_i, _, _, _, pi_given_s, _, G_bar in episode:

                # compute gradient
                phi_SelfRS_grad_accumulator[state_i, action_i] += 1.0 * \
                                                                    (pi_given_s[action_i] * (G_bar - self.V[state_i]))
                phi_SelfRS_grad_accumulator[state_i, :] += - pi_given_s[:] * \
                                                             (pi_given_s[action_i] * (G_bar - self.V[state_i]))

        # Update phi_ExploB
        self.phi_SelfRS += self.args["eta_phi_SelfRS"] * phi_SelfRS_grad_accumulator

        if self.args["use_clipping"]:
            self.phi_SelfRS[self.phi_SelfRS > self.args["clipping_epsilon"]] = self.args["clipping_epsilon"]
            self.phi_SelfRS[self.phi_SelfRS < -self.args["clipping_epsilon"]] = -self.args["clipping_epsilon"]
        # print(self.phi_SelfRS)

        # declare gradient for critic to do average update
        critic_grad = np.zeros(self.n_states)
        total_critic = 0.0
        # V update
        for episode in postprocessed_D:
            for state, _, _, _, _, _, _, G_bar in episode[::-1]:
                delta = (G_bar - self.V[state])
                critic_grad[state] += delta
                total_critic += 1.0

        # Update gradient critic
        self.V += self.args["eta_critic"] * critic_grad / total_critic
        # print(self.V)
        return
    # enddef

    def update_ExploRS(self, D):
        # # Update ExploB
        # self.update_ExploB(D)

        # Update SelfRS
        self.update_SelfRS(D)
    #enddef

    def update_sors(self, D):

        postprocessed_D = self.postprocess_data(D)

        # [i, j, label_{traj_i > traj_j}]
        pairwise_data = self.get_pairwise_data_using_return(postprocessed_D)

        # if there is no succesfull trajectory no update
        if len(pairwise_data) == 0:
            return

        # make pairwise data at most sors_n_pairs lengths
        if len(pairwise_data) <= self.args["sors_n_pairs"]:
            pairwise_data = pairwise_data
        else:
            pairwise_data_indexes = np.random.choice(range(0, len(pairwise_data)),
                                                     size=self.args["sors_n_pairs"])
            pairwise_data = np.array(pairwise_data, dtype=int)[pairwise_data_indexes]

        phi_sors_grad_accumulator = np.zeros((self.n_states, self.n_actions))
        for i, j in pairwise_data:

            episode_i = postprocessed_D[i]
            episode_j = postprocessed_D[j]

            # traj i
            phi_sors_grad_i = np.zeros((self.n_states, self.n_actions))
            return_r_phi_i = 0.0
            for t, (s_i, a_i, _, _, _, _, _, _) in enumerate(episode_i):
                phi_sors_grad_i[s_i, a_i] += (self.gamma ** t)
                return_r_phi_i += (self.gamma ** t) * self.phi_sors[s_i, a_i]

            # traj j
            phi_sors_grad_j = np.zeros((self.n_states, self.n_actions))
            return_r_phi_j = 0.0
            for t, (s_j, a_j, _, _, _, _, _, _) in enumerate(episode_j):
                phi_sors_grad_j[s_j, a_j] += (self.gamma ** t)
                return_r_phi_j += (self.gamma ** t) * self.phi_sors[s_j, a_j]

            #gradient
            phi_sors_grad_accumulator += - (1.0 - self.softmax_prob(return_r_phi_i, return_r_phi_j)) *\
                                         (phi_sors_grad_i - phi_sors_grad_j)

        self.phi_sors -= self.args["eta_phi_sors"] * phi_sors_grad_accumulator / len(pairwise_data)

        return
    #enddef

    def softmax_prob(self, a, b):
        return np.exp(a) / (np.exp(a) + np.exp(b))

    def postprocess_data(self, D):

        postprocessed_D = []

        # episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state)]
        for episode in D:
            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            postprocessed_epidata = self.get_postposessed_episode(self.env, episode)

            # add postprocessed episode
            postprocessed_D.append(postprocessed_epidata)

        return postprocessed_D
    #enddef

    def get_postposessed_episode(self, env_orig, episode):

        postprocessed_epidata = []
        for t in range(len(episode)):
            state, action, r_hat, next_state, G_hat, pi_given_s = episode[t]

            # get original reward
            r_bar = self.get_original_reward(env_orig, state, action)

            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            e_t = [state, action, r_hat, next_state, G_hat, pi_given_s, r_bar, 0.0]

            postprocessed_epidata.append(e_t)

        # compute return \bar{G} for every (s, a)
        G_bar = 0  # original return
        for i in range(len(postprocessed_epidata) - 1, -1, -1):  # iterate backwards
            _, _, _, _, _, _, r_bar, _ = postprocessed_epidata[i]
            G_bar = r_bar + env_orig.gamma * G_bar
            postprocessed_epidata[i][7] = G_bar  # update G_bar in episode

        if postprocessed_epidata[0][7] > 0.0 and self.first_succ_episode_number is None:
            self.first_succ_episode_number = copy.deepcopy(self.count_episode)
        else:
            self.count_episode += 1.0

        return postprocessed_epidata
    #enddef

    def get_original_reward(self, env_orig, state, action):
        r_bar = env_orig.reward[state, action]
        return r_bar
    #enddef

    def get_pairwise_data_using_return(self, postprocessed_D):
        pairwise_data = []

        for i, episode_i in enumerate(postprocessed_D):
            for j, episode_j in enumerate(postprocessed_D):
                G_bar_i = episode_i[0][7]
                G_bar_j = episode_j[0][7]
                if G_bar_i > G_bar_j:
                    # \tau_i > \tau_j
                    pairwise_data.append([i, j])

        return pairwise_data
    # enndef

    def indicator(self, state, action, s, a):
        return 1.0 if (state == s and action == a) else 0.0
# endclass
