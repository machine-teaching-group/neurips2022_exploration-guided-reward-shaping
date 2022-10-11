import numpy as np
import os
import copy

def sample_trajectory_given_state_action(env, agent, start_state, start_action=None,
                                         epsilon_reinforce=0.0, H=100):
    env_copy = copy.deepcopy(env)
    _ = env_copy.reset()
    env_copy.current_state = list(start_state)


    curr_state = start_state
    epidata = []
    iter = 0

    # rollout an entire episode
    while True:

        if start_action is not None and iter == 0:
            action = start_action
        else:
            # pick the action based on the demonstration
            action, _ = agent.predict(curr_state)
        # take a step
        next_state, reward, done, _ = env_copy.step(action)

        # get action distribution of the state
        # print(curr_state)
        pi_given_s = agent.get_action_distribution(curr_state)

        # store the data in epidata # e_t --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]
        e_t = [curr_state, action, reward, next_state, 0.0, pi_given_s]
        epidata.append(e_t)

        if done or H < iter:
            break

        # update curr_state
        curr_state = next_state
        iter += 1
    return epidata
#enddef

def generate_sampled_data(env_teacher, agent, epsilon_reinforce=0.0):

    # reset the environment: Get initial state and demonstration
    curr_state = env_teacher.reset()

    # episode data (epidata)
    epidata = [] # --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]

    # rollout an entire episode
    while True:

        # pick the action based on the demonstration
        action, _ = agent.predict(curr_state, epsilon=epsilon_reinforce)

        # take a step
        next_state, reward_hat, done, _ = env_teacher.step(action)

        # get action distribution of the state
        pi_given_s = agent.get_action_distribution(curr_state)

        # store the data in epidata # e_t --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]
        e_t = [curr_state, action, reward_hat, next_state, 0.0, pi_given_s]
        epidata.append(e_t)

        if done:
            break

        # update curr_state
        curr_state = next_state

    # compute \hat{G} for every (s, a)
    G_hat = 0  # shaped return
    for i in range(len(epidata) - 1, -1, -1):  # iterate backwards
        _, _, r_hat, _, _, _ = epidata[i]
        G_hat = r_hat + env_teacher.gamma * G_hat
        epidata[i][4] = G_hat  # update G_hat in episode

    return epidata
#enddef



# def get_policy_given_theta__(env, theta):
#     n_actions = env.n_actions
#     n_states = env.n_states
#     if theta.ndim == 1:
#         theta = theta.reshape(n_states, n_actions)
#     # this is for stable softmax (substract max)
#     theta = theta - np.repeat(np.max(theta, axis=1), n_actions).reshape(n_states, n_actions)
#
#     policy = np.exp(theta)/np.repeat(np.sum(np.exp(theta), axis=1),
#                             n_actions).reshape(n_states, n_actions)
#     return policy
# #enddef


def evaluate_agent(env_orig, env_teacher, agent, n_episode):

    episode_reward_array_env_orig = []
    episode_reward_array_env_teacher = []

    for i in range(n_episode):
        episode_reward_env_orig = 0
        episode_reward_env_teacher = 0
        episode = generate_sampled_data(env_teacher,  agent)

        for t in range(len(episode)):
            _, _, r_hat, _, _, _ = episode[t]

            # episode_reward_env_orig += env_orig.gamma**t * r_bar
            episode_reward_env_teacher += env_teacher.gamma**t * r_hat

        episode_reward_array_env_orig.append(episode_reward_env_orig)
        episode_reward_array_env_teacher.append(episode_reward_env_teacher)

    return np.average(episode_reward_array_env_orig), np.average(episode_reward_array_env_teacher)
#enddef


def write_into_file(accumulator, exp_iter, out_folder_name="out_folder", out_file_name="out_file"):
    directory = 'results/{}'.format(out_folder_name)
    filename = out_file_name + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef