import numpy as np
import torch
import os
import utils
import models as model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Agent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        # self.phi = args["phi"] #abstraction

        #declare actor network
        self.actor_network = model.ActorNetwork(env, args)
        ###### declare critic network
        # self.critic_network = model.CriticNetwork(env, args)
    #enddef


    def load_model(self, path_for_policy_network, path_for_value_network=None):
        '''
        :param env: for fixing critic network's parameters
        :param PATH: file path to critic network's saved parameters
        :return: critic network with the saved parameetrs
        '''
        self.actor_network.load_state_dict(torch.load(path_for_policy_network))
        # if path_for_value_network is not None:
        #     self.critic_network.load_state_dict(torch.load(path_for_value_network))

    # enddef

    def save_model(self, path_for_policy_network, path_for_value_network):
        '''
        :param network: Neural network
        :param PATH: file where network parameters are stored
        :return:-
        '''
        torch.save(self.actor_network.state_dict(), path_for_policy_network)
        # torch.save(self.critic_network.state_dict(), path_for_value_network)
    # enddef

    def forward_pass(self, state):
        state = np.array(state)
        '''
        :param state: np.array of batched/single state
        :return: np.array of action probabilities
        '''
        if state.ndim < 2:
            # exit()
            log_action_probs = self.actor_network.network(torch.Tensor(state).unsqueeze(0).float())
        else:  # for state batch
            log_action_probs = self.actor_network.network(torch.Tensor(state))

        action_probs = torch.exp(log_action_probs)

        return action_probs.squeeze(0).data.numpy()
    #enddef

    def predict(self, state, epsilon=0.0):
        action_probs = self.forward_pass(state)
        action_probs = (1-epsilon) * action_probs + (epsilon / len(action_probs))
        sampled_action = np.random.choice(np.array(len(action_probs)), p=action_probs)
        return sampled_action, action_probs
    #enddef


    def get_action_distribution(self, state):
        action_probs = self.forward_pass(state)
        return action_probs
    #enndef

    def update(self, D):

        '''
        :param D: np.array of batch of trajectories [(state, action, reward, next_state)]
        :param alpha: learning rate of Actor
        :param beta: learning rate of Critic
        :return: updates actor and critic of the agent
        '''

        ### accumulators
        loss_accumulator_array_policy = []
        loss_accumulator_array_value = []


        for traj in D:
            states_batch = []
            actions_batch = []
            returns_batch = []
            #TODO change this for loop
            for s, a, _, _, G, _ in traj:
                states_batch.append(s)
                actions_batch.append(a)
                # rewards_batch = np.array(traj)[:, 2]
                if torch.is_tensor(G):
                    returns_batch.append(G.detach().numpy())
                else:
                    returns_batch.append(G)

            states_batch = np.array(states_batch)
            actions_batch = np.array(actions_batch)
            returns_batch = np.array(returns_batch)

            # update the policy network
            loss = self.actor_network.update(states_batch, actions_batch, returns_batch)
            loss_accumulator_array_policy.append(loss)

            # # update value network
            # loss_critic = self.critic_network.update(states_batch, returns_batch)
            # loss_accumulator_array_value.append(loss_critic)
        return
    # enddef

    def logging_info(self, loss_accumulator_array_actor, loss_accumulator_array_critic, n_for_running_average,
                     curr_episode_number):

        # create path to save a model
        folder_path_actor = self.args["folderpath_model_actor"]
        folder_path_critic = self.args["folderpath_model_critic"]

        # check if folder exists
        if not os.path.exists(folder_path_actor):
            os.makedirs(folder_path_actor)
        # check if folder exists
        if not os.path.exists(folder_path_critic):
            os.makedirs(folder_path_critic)

        path_to_save_model_actor = "{}{}_{}".format(folder_path_actor, curr_episode_number, "model_AC.pt")
        path_to_save_model_critic = "{}{}_{}".format(folder_path_critic, curr_episode_number, "model_AC.pt")

        # compute mean of last n_running_average
        if (len(loss_accumulator_array_actor) == 0 or len(loss_accumulator_array_critic) == 0):
            mean_of_last_n_losses_actor = 100.0
            mean_of_last_n_losses_critic = 100.0
        # if(len(loss_accumulator_array_critic) == 0):
        #     mean_of_last_n_losses_critic = 100.0
        else:
            mean_of_last_n_losses_actor = np.mean(loss_accumulator_array_actor[-n_for_running_average:])
            mean_of_last_n_losses_critic = np.mean(loss_accumulator_array_critic[-n_for_running_average:])

        # save current model parameters
        self.save_model(path_to_save_model_actor, path_to_save_model_critic)

        # print information
        print("==================================================")
        print("Current Episode: {}".format(curr_episode_number))
        print("Mean of the last 100 losses Actor: {}".format(np.round(mean_of_last_n_losses_actor, 4)))
        print("Mean of the last 100 losses Critic: {}".format(np.round(mean_of_last_n_losses_critic, 4)))
        print("==================================================")
        return mean_of_last_n_losses_actor, mean_of_last_n_losses_critic
    # enddef

if __name__ == "__main__":
    pass

