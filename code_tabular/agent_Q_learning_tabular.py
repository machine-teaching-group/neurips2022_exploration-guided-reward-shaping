import numpy as np
import copy


class Agent:
    def __init__(self, env, args):

        self.env = env
        self.args = args

        # Q function
        self.Q = np.zeros((env.n_states, env.n_actions))

        _, pi_stoch_epsilon_greedy = self.get_epsilon_greedy_policies_given_Q(self.Q, epsilon=self.args["Q_epsilon"])
        self.policy = pi_stoch_epsilon_greedy


    # enddef

    def update(self, D):
        '''
        :param episode: np.array of batch of trajectories [(state, action, reward, next_state)]
        :return: updates and critic of the agent
        '''

        # iterate every episode
        for episode in D:

            # Q update
            for s, a, r_hat, s_n, _, _ in episode[::-1]: #reverse ordering
                # TD Update
                best_next_action = np.argmax(self.Q[s_n])
                td_target = r_hat + self.env.gamma * self.Q[s_n][best_next_action]
                td_delta = td_target - self.Q[s][a]
                self.Q[s][a] = self.Q[s][a] + self.args["Q_alpha"] * td_delta

        # policy update
        _, pi_stoch_epsilon_greedy = self.get_epsilon_greedy_policies_given_Q(self.Q, epsilon=self.args["Q_epsilon"])
        self.policy = pi_stoch_epsilon_greedy

        return
    # enddef

    def predict(self, state):
        '''
        :param state: int
        :return: action and np.array of action probabilities
        '''

        action_probs = self.policy[state]
        sampled_action = np.random.choice(range(0, len(action_probs)), p=action_probs)
        return sampled_action, action_probs
    # enddef

    def get_action_distribution(self, state):
        return self.policy[state]
    # enddef

    def get_epsilon_greedy_policies_given_Q(self, Q, epsilon=0.1, tol=1e-6):

        n_states = Q.shape[0]
        n_actions = Q.shape[1]

        pi_det = np.argmax(Q, axis=1)

        # Get a stochastic policy
        pi_s = Q - np.max(Q, axis=1)[:, None]
        pi_s[np.where((-tol <= pi_s) & (pi_s <= tol))] = 1
        pi_s[np.where(pi_s <= 0)] = 0
        pi_s = pi_s / pi_s.sum(axis=1)[:, None]

        #compute pi_det_epsilon_greedy
        pi_det_epsilon_greedy = np.ones((n_states, n_actions)) * epsilon / n_actions
        pi_det_epsilon_greedy[range(n_states), pi_det] += (1.0 - epsilon)

        #compute pi_stochastic_epsilon_greedy
        pi_stochastic_epsilon_greedy = copy.deepcopy(pi_s)
        pi_stochastic_epsilon_greedy *= (1.0 - epsilon)
        pi_stochastic_epsilon_greedy += epsilon/n_actions

        return pi_det_epsilon_greedy, pi_stochastic_epsilon_greedy
    #enddef




if __name__ == "__main__":
    pass



