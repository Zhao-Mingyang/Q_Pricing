import torch
import torch.nn as nn
import numpy as np
from random import sample

class QNetwork(nn.Module):
    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):

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
        self.activation = nn.Softmax(dim=-1)

    def forward(self, s):

        # print(s.size())
        x = self.flatten(s)
        x = self.normalisation(x)
        x = self.fc1(x)
        x = self.actionvation(x)
        x= self.fc2(x)
        x = self.actionvation(x)
        out = self.activation(self.fc3(x))
        return out

class adj_QNetwork(nn.Module):
    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):

        super(adj_QNetwork, self).__init__()

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
        self.activation = nn.Softmax(dim=-1)

    def forward(self, s, oppo_s):
        # print(s.size())
        x = self.flatten(s)
        # print(x.size())
        x = self.normalisation(x)
        x = self.fc1(x)
        x = self.actionvation(x)
        x= self.fc2(x)
        x = self.actionvation(x)
        out = self.activation(self.fc3(x))

        oppo_x = self.flatten(oppo_s)
        oppo_x = self.normalisation(oppo_x)
        oppo_x = self.fc1(oppo_x)
        oppo_x = self.actionvation(oppo_x)
        oppo_x = self.fc2(oppo_x)
        oppo_x = self.actionvation(oppo_x)
        oppo_out = self.activation(self.fc3(oppo_x))

        return (out + oppo_out)/2

class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, price, action, reward):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = [state, price, action, reward]
        self.memory.append(transition)

    def pop_batch(self):

        samples = sample(self.memory, self.batch_size)
        # samples = self.memory[-self.batch_size:]
        # print(len(samples))
        # self.memory = self.memory
        # print(len(self.memory))
        return map(np.array, zip(*samples))

class adj_Memory(object):

    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.num_price = 100

    def save(self, state, oppo_state, price, action, reward):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        prices = [price] * self.num_price
        transition = [state, oppo_state, prices, action, reward]
        self.memory.append(transition)

    def pop_batch(self):

        samples = sample(self.memory, self.batch_size)
        # samples = self.memory[-self.batch_size:]
        # print(len(samples))
        # self.memory = self.memory
        # print(len(self.memory))
        return map(np.array, zip(*samples))