import numpy as np
import random
from Q_network import QNetwork
import torch
import torch.nn as nn
from Q_network import Memory

class Agent():
    def __init__(self, memory_size = 5000, batch_size = 2048, cheating =  True):
        self.num_price = 100
        self.prices = np.round(np.arange(1/self.num_price, 1.01, 1/self.num_price), 3)
        if cheating:
            self.q_value = np.array([[max(0.48 - 0.9 * myprice + 0.6 * oppoprice, 0) for myprice in self.prices]for oppoprice in self.prices])
        else:
            self.q_value = np.ones([self.num_price,self.num_price])

        self.q_nums = np.ones([self.num_price, self.num_price])
        self.mean_q_value = np.zeros(self.num_price)
        self.EPSILON= 0.5
        self.ALPHA = 1
        self.step_size = 7e-7
        self.maxmean = False
        self.itersteps = 0
        self.decision_range = 40
        self.KLlossWeight = 0.1
        self.RlossWeight = 0.5

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mlp_layers = [256, 256]
        self.q_policy = QNetwork(num_actions=self.num_price, state_shape=(self.num_price * 2 + 1), \
                                     mlp_layers=self.mlp_layers).to(device=self.device)
        self.q_policy.eval()
        self.optimizer = torch.optim.Adam(self.q_policy.parameters(), lr=5e-4)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.memory = Memory(memory_size, batch_size)
        self.batch_size = batch_size
        self.update_freq = 1

    def update(self, myprice, oppoprice, reward):
        myindex = np.where(self.prices ==  myprice)[0][0]
        oppoindex = np.where(self.prices ==  oppoprice)[0][0]
        self.q_nums[myindex][oppoindex] += 1
        self.q_value[myindex][oppoindex] = (1-self.ALPHA)*self.q_value[myindex][oppoindex] + self.ALPHA * reward \
                                           # + self.ALPHA * np.random.uniform(-25/1000, 25/1000)
        # if self.ALPHA > 0.05:
        #     self.ALPHA -= self.step_size
        if self.maxmean == False:
            if self.EPSILON < 1:
                self.EPSILON += self.step_size

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
        return index

    def neural_estimate(self):
        theindex = self.maxexpected_policy()
        estimated_q = []
        index_q = []
        for i in range(max(0, theindex - self.decision_range), min(self.num_price, theindex + self.decision_range + 1)):
            price_count = np.sum(self.q_nums[i])
            q_freq = self.q_nums[i] / price_count
            s = np.concatenate((self.q_value[i], q_freq), axis=None)
            s = np.append(s, self.prices[i])
            s = torch.unsqueeze(torch.FloatTensor(s), dim=0).to(self.device)
            p_action = self.q_policy(s).detach().cpu().numpy()
            # print(p_action, np.argmax(p_action))
            the_value = np.sum(self.q_value[i] * p_action)
            index_q.append(i)
            estimated_q.append(the_value)
        index_q = np.array(index_q)
        estimated_q = np.array(estimated_q)
        indexmax = np.where(estimated_q == np.amax(estimated_q))[0]
        # print(indexmax)
        action = random.choice(index_q[indexmax])
        return self.prices[action]

    def eplison_greedy(self):
        self.itersteps += 1
        if np.random.uniform() < self.EPSILON:
            if self.maxmean:
                return self.prices[self.maxexpected_policy()]
            else:
                if self.itersteps < self.num_price * self.num_price * 5:
                    return self.prices[self.maxexpected_policy()]
                else:
                    return self.neural_estimate()
        else:
            return random.choice(self.prices)

    def feed(self, memory):
        reward_bias = 1e-6
        s, price, a, r = tuple(memory)
        self.memory.save(s, price, a, r+reward_bias)

        if self.itersteps % self.update_freq == 0 and self.itersteps > self.batch_size:
            batch = self.memory.pop_batch()
            s, price, a, r = batch
            # print(s[0], price[0], a[0], r[0])
            s_prob = s[:,self.num_price:]
            # print(s_prob.shape)
            price = np.expand_dims(price, 1)
            # print(s)
            # print(s.shape, price.shape)
            s = np.concatenate((s, price), axis=1)
            s = torch.FloatTensor(s).to(self.device)
            s_prob = torch.FloatTensor(s_prob).to(self.device)
            a = torch.FloatTensor(a).to(self.device).long()
                # .unsqueeze(1)
            # print(a.size())
            r = torch.FloatTensor(r).to(self.device).unsqueeze(1)

            self.q_policy.train()
            q_act = self.q_policy(s)
            # print(q_act.size())
            Rloss = torch.mean(torch.log10(r))
            CEloss = self.loss(q_act, a)
            KLloss = self.kl_loss(q_act,s_prob)
            # print('celoss',CEloss,'klloss', KLloss,'Rloss', Rloss)
            loss = CEloss + self.KLlossWeight*KLloss + self.RlossWeight * Rloss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.q_policy.eval()
