import numpy as np
import utils
from scipy.special import softmax
import copy


class Agent:
    def __init__(self, env, args):

        self.env = env
        self.args = args

        # actor parameter theta
        self.actor_theta = np.zeros((env.n_states, env.n_actions))
        # declare actor policy
        self.actor_policy = self.get_policy_given_theta(env, self.actor_theta)
    # enddef

    def update(self, D):
        '''
        :param episode: np.array of batch of trajectories [(state, action, reward, next_state)]
        :return: updates actor
        '''

        total_actor = 0
        actor_theta_grad = np.zeros((self.env.n_states, self.env.n_actions))
        # iterate every episode
        for episode in D:

            # Actor theta parameter update
            for s, a, r_hat, s_n, G_hat, _ in episode:
                actor_theta_grad[s, a] = actor_theta_grad[s, a] + G_hat
                actor_theta_grad[s, :] = actor_theta_grad[s, :] - G_hat * self.actor_policy[s, :]
                total_actor += 1.0

        # Compute gradient actor
        self.actor_theta += self.args["eta_actor"] * actor_theta_grad / total_actor

        # Actor policy update
        self.actor_policy = self.get_policy_given_theta(self.env, self.actor_theta)


        return
    # enddef

    def predict(self, state):
        '''
        :param state: int
        :return: action and np.array of action probabilities
        '''

        action_probs = self.actor_policy[state]
        sampled_action = np.random.choice(range(0, len(action_probs)), p=action_probs)
        return sampled_action, action_probs
    # enddef

    def get_action_distribution(self, state):
        return self.actor_policy[state]
    # enddef

    def get_policy_given_theta_old(self, env, theta):
        n_actions = env.n_actions
        n_states = env.n_states
        if theta.ndim == 1:
            theta = theta.reshape(n_states, n_actions)

        return np.nan_to_num(softmax(theta, axis=1))
    #enddef

    def get_policy_given_theta(self, env, theta, axis=-1):

        n_actions = env.n_actions
        n_states = env.n_states
        if theta.ndim == 1:
            theta = theta.reshape(n_states, n_actions)

        xs = theta - np.max(theta, axis=axis, keepdims=True)
        xs_exp = np.nan_to_num(np.exp(xs))
        xs_exp = np.nan_to_num(xs_exp)
        return np.nan_to_num(xs_exp / xs_exp.sum(axis=axis, keepdims=True))
    #enddef

