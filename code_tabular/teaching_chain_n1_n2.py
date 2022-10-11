import numpy as np
import env_chain_n1_n2
import agent_reinforce_tabular as agent_reinforce
import agent_Q_learning_tabular as agent_Q_learning
import env_generic_with_teacher as Teacher_env
import utils
import plot_results

from collections import deque as collections_deque
import parameters
import argparse

import sys


class teaching():
    # hyperparameters
    hyper_parameters = parameters.parameters()
    teacher_args = hyper_parameters.teacher_args_chain
    agent_args = hyper_parameters.agent_args
    teaching_args = hyper_parameters.teaching_args

    def __init__(self, env_orig, teacher_name="", agent_type=""):
        self.env_orig = env_orig
        self.teacher_name = teacher_name
        self.agent_type = agent_type
        self.env_teacher = None

        if agent_type == "reinforce":
            ## use no clipping for reinforce
            self.teacher_args["use_clipping"] = False
            self.env_teacher = Teacher_env.EnvTeacher(env_orig, self.teacher_args, teacher_name)
            ## define agent
            self.agent = agent_reinforce.Agent(self.env_teacher, self.agent_args)
            self.teaching_args["N"] = self.teaching_args["N_reinforce"]
        elif agent_type == "Q_learning":
            ## use clipping for Q_learning
            self.teacher_args["use_clipping"] = True
            self.env_teacher = Teacher_env.EnvTeacher(env_orig, self.teacher_args, teacher_name)
            ## define agent
            self.agent = agent_Q_learning.Agent(self.env_teacher, self.agent_args)
            self.teaching_args["N"] = self.teaching_args["N_Q_learning"]
        else:
            print(" Wrong agent type")
            exit(0)
        self.accumulator = {}
    #enddef

    def end_to_end_training(self):


        N = self.teaching_args["N"]
        N_r = self.teaching_args["N_r"]
        N_p = self.teaching_args["N_p"]
        buffer_size = self.teaching_args["buffer_size"]
        buffer_size_recent = self.teaching_args["buffer_size_recent"]
        agent_evaluation_step = self.teaching_args["agent_evaluation_step"]

        D = collections_deque(maxlen=buffer_size)

        expected_reward_array_env_orig = []
        expected_reward_arrray_env_teacher = []

        for i in range(N):

            if i % agent_evaluation_step == 0:

                # evaluate learner's current policy on orig environment
                expected_reward_G_bar, expected_reward_G_hat = self.evaluate_agent(self.env_orig, self.env_teacher, self.agent,
                                                                                    n_episode=5)
                expected_reward_array_env_orig.append(expected_reward_G_bar)
                expected_reward_arrray_env_teacher.append(expected_reward_G_hat)

                print("===============Iter = {}/{}========================".format(i, N))
                print("Teacher = {}".format(self.teacher_name))
                print("Exp reward = {}".format(np.round(expected_reward_array_env_orig[-1], 4)))
                print("Goal visitation count = {}".format(self.env_teacher.goal_visits))
                print("===================================================")

            # rollout a trajectory --> [state, action, r, next_state, G_r,  \pi(.|state)] r = \hat(r) on teacher's env
            episode = utils.generate_sampled_data(self.env_teacher, self.agent)  # --> rewrite

            # add to buffer
            D.append(episode)

            # teacher update
            if (i + 1) % N_r == 0:

                # print("=== Teacher update ===")

                self.env_teacher.update(D,  self.agent)

            # learner's update
            if (i + 1) % N_p == 0:

                # print("=== Learner update ===")

                self.agent.update(list(D)[-buffer_size_recent:])

        self.accumulator["expected_reward_env_orig_{}".format(self.env_teacher.teacher_name)] = \
            np.array(expected_reward_array_env_orig)
        self.accumulator["expected_reward_env_teacher_{}".format(self.env_teacher.teacher_name)] = \
            np.array(expected_reward_arrray_env_teacher)
        # self.accumulator["first_succ_episode_number_teacher_{}".format(self.env_teacher.teacher_name)] = \
        #     [self.env_teacher.first_succ_episode_number]

        return self.accumulator
    #enddef

    def evaluate_agent(self, env_orig, env_teacher, agent, n_episode):
        episode_reward_array_env_orig = []
        episode_reward_array_env_teacher = []

        for i in range(n_episode):
            episode_reward_env_orig = 0
            episode_reward_env_teacher = 0
            episode = utils.generate_sampled_data(env_teacher, agent)
            postProcessed_episode = self.env_teacher.get_postposessed_episode(env_orig, episode)

            for t in range(len(postProcessed_episode)):
                _, _, r_hat, _, _, _, r_bar, _ = postProcessed_episode[t]

                episode_reward_env_orig += env_orig.gamma**t * r_bar
                episode_reward_env_teacher += env_teacher.gamma ** t * r_hat

            episode_reward_array_env_orig.append(episode_reward_env_orig)
            episode_reward_array_env_teacher.append(episode_reward_env_teacher)

        return np.average(episode_reward_array_env_orig), np.average(episode_reward_array_env_teacher)
    #enddef

#endclass

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef
def accumulator_function(tmp_dict, dict_accumulator):
    for key in tmp_dict:
        if key in dict_accumulator:
            dict_accumulator[key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef


if __name__ == "__main__":

    final_dict_accumulator = {}

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--agent', default='reinforce', type=str, help='agent type')
    parser.add_argument('--n2_subgoal', default=0.00, type=float, help='n2 subgoal')
    parser.add_argument('--n_averaged', default=10, type=int, help='N runs to average')


    args = parser.parse_args()
    agent_type = args.agent
    n2_subgoal = args.n2_subgoal #0.05
    n2_subgoal_position = 15

    output_directory = "runs_chain/agent={}_n2_subgoal={}".format(args.agent, args.n2_subgoal)
    path = "results/{}".format(output_directory)
    out_folder_name = "results/plots"

    if args.agent == "Q_learning":
        teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "SORS_with_Rbar"]
    elif args.agent == "reinforce":
        teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "SORS_with_Rbar", "LIRPG_without_metagrad"]
    else:
        print("Agent type should be either Q_learning or reinforce")
        print("agent type: {}".format(args.agent))
        exit()

    for i in range(0, args.n_averaged):

        for n1 in [20]:
            for n2 in [40]:

                env_args = {
                    "Rmax": 1,
                    "n_1": n1,
                    "n_2": n2,
                    "n_2_subgoal": n2_subgoal,
                    "n2_subgoal_position": n2_subgoal_position,
                    "n_actions:": 2,
                    "gamma": 0.99,
                    "randomMoveProb": 0.05,
                    "H": 40
                }
                env_orig = env_chain_n1_n2.ChainEnvironment(env_args)

                dict_accumulator = {}
                for teacher_name in teachers:

                    teaching_obj = teaching(env_orig, teacher_name, agent_type)

                    dict_acc = teaching_obj.end_to_end_training()
                    dict_accumulator.update(dict_acc)

                utils.write_into_file(accumulator=dict_accumulator, exp_iter=i + 1,
                                      out_folder_name=output_directory)

    if args.agent == "Q_learning":
        fig_name = "chain_wo_distractor_qlearning" if args.n2_subgoal == 0.0 else "chain_with_distractor_qlearning"
        plot_results.plot_Q_learning_curves(path, out_folder_name, fig_name,  n_file=args.n_averaged, t=330)

    else:
        fig_name = "chain_wo_distractor_reinforce" if args.n2_subgoal == 0.0 else "chain_with_distractor_reinforce"
        plot_results.plot_reinforce_curves(path, out_folder_name, fig_name, n_file=args.n_averaged, t=330)

