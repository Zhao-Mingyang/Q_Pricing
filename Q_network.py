import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(QNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        self.flatten = nn.Flatten()
        self.normalisation = nn.BatchNorm1d(layer_dims[0])
        self.fc1 = nn.Linear(layer_dims[0], layer_dims[1], bias=True)
        # self.actionvation = nn.ReLU()
        self.actionvation = nn.Tanh()
        self.fc2 = nn.Linear(layer_dims[1], layer_dims[2], bias=True)
        self.fc3 = nn.Linear(layer_dims[2], self.num_actions, bias=True)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        print(s.size())
        x = self.flatten(s)
        x = self.normalisation(x)
        x = self.fc1(x)
        x = self.actionvation(x)
        x= self.fc2(x)
        x = self.actionvation(x)
        out = self.fc3(x)
        return out

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = [state, action, reward]
        self.memory.append(transition)

    def pop_batch(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = self.memory[-self.batch_size:]
        # print(samples)
        # self.memory = self.memory
        print(len(self.memory))
        return map(np.array, zip(*samples))