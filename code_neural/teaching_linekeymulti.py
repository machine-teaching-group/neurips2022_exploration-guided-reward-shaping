import numpy as np
import env_linekeymulti
import agent_reinforce as agent_reinforce
import env_generic_with_teacher as Teacher_env
import utils
import plot_results

from collections import deque as collections_deque
import parameters
import argparse

import sys
import torch


class teaching():
    # hyperparameters
    hyper_parameters = parameters.parameters()
    teacher_args = hyper_parameters.teacher_args_chain
    agent_args = hyper_parameters.agent_args
    teaching_args = hyper_parameters.teaching_args

    def __init__(self, env_orig, teacher_name="", agent_type="", epsilon_reinforce=0.0, n_picks=1, n_actions=1):
        self.env_orig = env_orig
        self.teacher_name = teacher_name
        self.agent_type = agent_type
        self.env_teacher = Teacher_env.EnvTeacher(env_orig, self.teacher_args, teacher_name)
        self.epsilon_reinforce = epsilon_reinforce
        self.n_picks = n_picks
        self.n_actions = n_actions

        self.agent = agent_reinforce.Agent(self.env_teacher, self.agent_args)
        self.accumulator = {}
    #enddef

    def end_to_end_training(self):


        N = self.teaching_args["N_reinforce"]
        N_r = self.teaching_args["N_r"]
        N_p = self.teaching_args["N_p"]
        buffer_size = self.teaching_args["buffer_size"]
        buffer_size_recent = self.teaching_args["buffer_size_recent"]
        agent_evaluation_step = self.teaching_args["agent_evaluation_step"]
        # provide_shaped_reward_after_n_steps = self.teaching_args["provide_shaped_reward_after_n_steps"]

        D = collections_deque(maxlen=buffer_size)

        expected_reward_array_env_orig = []
        expected_reward_arrray_env_teacher = []

        for i in range(N):

            # switch when n_steps_to_follow_orig has been reached
            if self.env_teacher.n_steps_to_follow_orig < i:
                self.env_teacher.switch_teacher = True


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
                # print(np.round(self.env_teacher.rexplore_w, 4))

            # rollout a trajectory --> [state, action, r, next_state, G_r,  \pi(.|state)] r = \hat(r) on teacher's env
            episode = utils.generate_sampled_data(self.env_teacher, self.agent, self.epsilon_reinforce)  # --> rewrite

            #%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%
            # add to buffer
            D.append(episode)

            # teacher update
            if (i + 1) % N_r == 0:

                # print("=== Teacher update ===")

                self.env_teacher.update(D, self.agent, self.epsilon_reinforce)

            # learner's update
            if (i + 1) % N_p == 0:

                # print("=== Learner update ===")

                self.agent.update(list(D)[-buffer_size_recent:])

        self.accumulator["expected_reward_env_orig_{}".format(self.env_teacher.teacher_name)] = \
            np.array(expected_reward_array_env_orig)
        self.accumulator["expected_reward_env_teacher_{}".format(self.env_teacher.teacher_name)] = \
            np.array(expected_reward_arrray_env_teacher)

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

        # episode_reward_array_env_orig = np.array(episode_reward_array_env_orig)
        # episode_reward_array_env_teacher = np.array(episode_reward_array_env_teacher)
        R_orig_s = torch.Tensor.mean(torch.Tensor(episode_reward_array_env_orig))
        R_shaped_s = torch.mean(torch.Tensor(episode_reward_array_env_teacher))
        return R_orig_s.detach().numpy(), \
               R_shaped_s.detach().numpy()
    #enddef

#endclass




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--init_state', type=str, default="[0.3, 0.4]")
    parser.add_argument('--finish_action_termination_flag', default="False", type=str, help='Flag for finish action to terminate')
    parser.add_argument('--small_reward_for_picking_key', default=0.0, type=float, help='Small reward for picking key')
    parser.add_argument('--small_reward_for_goal_without_key', default=0.0, type=float, help='Small reward for goal with no key')
    parser.add_argument('--random_small_reward_for_goal_without_key', default=0.0, type=float, help='Random small reward for goal with no key')

    parser.add_argument('--epsilon_reinforce', default=0.0, type=float, help='Epsilon for reinforce algorithm')
    parser.add_argument('--teacher', default="", type=str, help='Teacher name')
    parser.add_argument('--n_picks', default=1, type=int, help='Number of picks')
    parser.add_argument('--n_averaged', default=10, type=int, help='Number of averaged')

    args = parser.parse_args()

    init_state = eval(args.init_state)
    finish_action_termination_flag = eval(args.finish_action_termination_flag)
    small_reward_for_picking_key = args.small_reward_for_picking_key
    random_small_reward_for_goal_without_key = args.random_small_reward_for_goal_without_key
    small_reward_for_goal_without_key = args.small_reward_for_goal_without_key
    epsilon_reinforce = args.epsilon_reinforce
    teacher = args.teacher
    n_picks = args.n_picks
    H = 60

    n_actions = 3 if n_picks == 1 else 4

    teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "SORS_with_Rbar", "LIRPG_without_metagrad"]

    output_directory = "runs_lineKey/agent={}_small_reward_for_goal_without_key={}".format("reinforce",
                                                                                           args.small_reward_for_goal_without_key)
    path = "results/{}".format(output_directory)
    out_folder_name = "results/plots"

    for i in range(0, args.n_averaged):
        env_args = {
            "R_max": 1,
            "gamma": 0.99,
            "randomMoveProb": 0.05,
            "n_picks": n_picks,
            "n_actions": n_actions,
            "H": H,
            "init_state": init_state,
            "finish_action_termination_flag": finish_action_termination_flag,
            "small_reward_for_picking_key": small_reward_for_picking_key,
            "small_reward_for_goal_without_key": small_reward_for_goal_without_key,
            "random_small_reward_for_goal_without_key": random_small_reward_for_goal_without_key
        }

        env_orig = env_linekeymulti.Environment(env_args)

        dict_accumulator = {}

        for teacher_name in teachers:
            teaching_obj = teaching(env_orig, teacher_name, agent_type="reinforce",
                                    epsilon_reinforce=epsilon_reinforce, n_picks=n_picks, n_actions=n_actions)

            dict_acc = teaching_obj.end_to_end_training()
            dict_accumulator.update(dict_acc)

        utils.write_into_file(accumulator=dict_accumulator, exp_iter=i + 1,
                              out_folder_name=output_directory)

    fig_name = "linekey_wo_distractor_reinforce" if args.small_reward_for_goal_without_key == 0.0 else "linekey_with_distractor_reinforce"
    plot_results.plot_reinforce_curves(path, out_folder_name, fig_name, n_file=args.n_averaged, t=510)
