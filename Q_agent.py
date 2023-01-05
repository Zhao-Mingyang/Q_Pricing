import numpy as np
import random
from Q_network import QNetwork
import torch

class Agent():
    def __init__(self):
        self.num_price = 100
        self.prices = np.round(np.arange(1/self.num_price, 1.01, 1/self.num_price), 3)

        self.q_value = np.ones([self.num_price,self.num_price])
        self.q_nums = np.ones([self.num_price, self.num_price])
        self.mean_q_value = np.zeros(self.num_price)
        self.EPSILON= 0.9
        self.ALPHA = 1
        self.step_size = 1e-7
        self.maxmean = True

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mlp_layers = [128, 128]
        self.q_policy = QNetwork(num_actions=self.num_price, state_shape=self.num_price * self.num_price, \
                                     mlp_layers=self.mlp_layers).to(device=self.device)
        self.q_policy.eval()
        self.optimizer = torch.optim.Adam(self.q_policy.parameters(), lr=5e-4)

    def update(self, myprice, oppoprice, reward):
        myindex = np.where(self.prices ==  myprice)[0][0]
        oppoindex = np.where(self.prices ==  oppoprice)[0][0]
        self.q_nums[myindex][oppoindex] += 1
        self.q_value[myindex][oppoindex] = (1-self.ALPHA)*self.q_value[myindex][oppoindex] + self.ALPHA * reward \
                                            + self.ALPHA * np.random.uniform(-75/1000, 75/1000)
        if self.ALPHA > 0.05:
            self.ALPHA -= self.step_size

    def maxexpected_policy(self):
        total_count = np.sum(self.q_nums, axis = 1)
        total_count = np.reshape(total_count, (-1, 1))
        # print('total_count',total_count)
        # print('self.q_nums', self.q_nums)
        # print('prob',np.divide(self.q_nums, total_count))
        # print(self.q_value * np.divide(self.q_nums, total_count))
        self.mean_q_value = np.sum(self.q_value * np.divide(self.q_nums, total_count), axis = 1)
        # print(self.mean_q_value)
        # for the_price in range(self.num_price):
        #     # self.mean_q_value[the_price] = 0
        #     # self.mean_q_value[the_price] = np.mean(self.q_value[the_price])
        #     total_count = np.sum(self.q_nums[the_price])
        #     # for oppo_price in range(self.num_price):
        #     self.mean_q_value[the_price] = np.sum(self.q_value[the_price]*self.q_nums[the_price]/total_count)
        # print(self.mean_q_value)
        index = np.where(self.mean_q_value == np.amax(self.mean_q_value))
        index = random.choice(index[0])
        return self.prices[index]

    def eplison_greedy(self):
        if np.random.uniform() < self.EPSILON:
            if self.maxmean:
                return self.maxexpected_policy()
            else:
                s = self.q_value.flatten()
                s = torch.unsqueeze(torch.FloatTensor(s), dim = 0).to(self.device)
                action_value = self.q_policy(s)
                action = torch.max(action_value, dim=1)[1].item()

                return self.prices[action]
        else:
            return random.choice(self.prices)

    def feed(self, memory):
        s, a, r = tuple(memory)
        s = torch.FloatTensor(s).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)

        self.q_policy.train()
        q_act = self.q_policy(s)

        dist = torch.distributions.Categorical(q_act)
        log_probs = dist.log_prob(a).view(-1, 1)
        loss = -(log_probs * (r-q_act)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        self.q_policy.eval()
