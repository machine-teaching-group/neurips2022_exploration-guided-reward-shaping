import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ActorNetwork(torch.nn.Module):
    def __init__(self, env, argv, n_nodes=256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_nodes, env.action_space.n),
            torch.nn.LogSoftmax(dim=-1)
        ).float()
        # optimizer for the Actor Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=argv["eta_actor"])
        #enddef

    # returns the action-probability distribution of a state, in numpy form: \pi(.|state)
    def predict(self, state):
        '''
        :param state: np.array of batched/single state
        :return: np.array of action probabilities
        '''
        if state.ndim < 2:
            log_action_probs = self.network(torch.FloatTensor(state).unsqueeze(0).float())
        else:  # for state batch
            log_action_probs = self.network(torch.FloatTensor(state))

        action_probs = torch.exp(log_action_probs)

        return action_probs
    #enddef

    def update(self, states, actions, returns):
        '''
        :param states: np.array of batched states
        :param actions: np.array of batched actions
        :param returnss: np.array of batched returnss
        :return: -- performs 1 actor network update
        '''
        state_batch = torch.Tensor(states)
        action_batch = torch.Tensor(actions)
        returns_batch = torch.Tensor(returns)

        # obtain the log_action-probabilities for each state in the episode from the network
        # pred_batch contains the gradient information for the states
        log_pred_batch = self.network(state_batch)

        # log_prob_batch contains gradient information for the state
        log_prob_batch = log_pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
        loss_tensor = log_prob_batch * returns_batch

        # compute the loss for batch
        loss = -torch.mean(loss_tensor)
        loss_to_return = loss

        # back-propagate the loss
        self.optimizer.zero_grad()
        loss.backward()

        # update the parameters of the Actor Network
        self.optimizer.step()
        return loss_to_return.detach().numpy()
    #enddef
# endclass


#critic class
class CriticNetwork(torch.nn.Module):
    def __init__(self, env, argv, n_nodes=256):
        super().__init__()
        # specifics of the network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_nodes, 1)
        ).float()
        # optimizer for the Critic Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=argv["eta_critic"])



    # returns the state-value of a state, in numpy form: V(state)
    def predict(self, state):
        '''
        :param state: np.array of batched/single state
        :return: np.array of action probabilities
        '''
        if state.ndim < 2:
            values = self.network(torch.FloatTensor(state).unsqueeze(0).float())
        else:  # for state batch
            values = self.network(torch.FloatTensor(state))

        return values
    #enddef

    def update(self, states, targets):
        '''
        :param states: np.array of batched states
        :param targets: np.array of values
        :return: -- # performs 1 update on Critic Network
        '''
        state_batch = torch.Tensor(states)
        target_batch = torch.Tensor(targets)

        # obtain the pred values for the states in the episode
        pred_batch = self.network(state_batch)

        # compute the MSE loss for the critic network based on the batch
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, target_batch.unsqueeze(1))

        # back-propagate the loss
        self.optimizer.zero_grad()
        loss.backward()

        # update the parameters of the Critic Network
        self.optimizer.step()

        return loss.detach().numpy()
    #enddef
#endclass


class RexploitNetwork(torch.nn.Module):
    def __init__(self, env, argv, n_nodes=256):
        super().__init__()
        # specifics of network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_nodes, env.action_space.n),
            torch.nn.Tanh()
        ).float()
        # optimizer for the Actor Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=argv["eta_phi_SelfRS"])

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):  # initialize with ones
    #     if isinstance(module, torch.nn.Linear):
    #         module.weight.data.fill_(0.0)
    #
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)
    #     # enddef
#endclass

class RSORSNetwork(torch.nn.Module):
    def __init__(self, env, argv, n_nodes=256):
        super().__init__()
        # specifics of network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_nodes, env.action_space.n),
            torch.nn.Tanh()
        ).float()
        # optimizer for the Actor Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=argv["eta_phi_sors"])

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):  # initialize with ones
    #     if isinstance(module, torch.nn.Linear):
    #         module.weight.data.fill_(0.0)
    #
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)
    #     # enddef

    # enddef
# endclass


class RLIRPGNetwork(torch.nn.Module):
    def __init__(self, env, argv, n_nodes=256):
        super().__init__()
        # specifics of network architecture
        self.network = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_nodes, env.action_space.n),
            torch.nn.Tanh()
        ).float()
        # optimizer for the Actor Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=argv["eta_phi_lirpg"])

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):  # initialize with ones
    #     if isinstance(module, torch.nn.Linear):
    #         module.weight.data.fill_(0.0)
    #
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)
    #     # enddef

    # enddef
# endclass


