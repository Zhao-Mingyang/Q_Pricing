from Q_agent import Agent
from env import Env
import torch
import numpy as np
from utils import *
import random

file_path = 'C:/Users/13862/Desktop/Pricing/experiments/'
time = 0.4
adj_smooth = False
continueous_price = True

env = Env()
agents = [Agent(adj_smooth = adj_smooth, continueous_price = continueous_price) for i in range(env.num_firms)]
decimals = agents[0].decimals
print(agents[0].q_value)
T = int(500000 * 1.75)
price_record = [[0 for i in range(T)] for _ in range(env.num_firms)]
for t in range(T):
    prices = [0 for i in range(env.num_firms)]
    for i in range(env.num_firms):
        prices[i] = agents[i].eplison_greedy()
        price_record[i][t] = prices[i]
    # if t>agents[i].num_price*agents[i].num_price * 7:
    #     if np.random.uniform() < agents[1].EPSILON:
    #         prices[1] = 0.4
    #     else:
    #         prices[1] = random.choice(agents[1].prices)
        # print(agents[1].q_value)
    rewards = env.reward(prices)

    print(prices, rewards)
    for i in range(env.num_firms):
        if continueous_price:
            theprice_index = int(round(prices[i], decimals) * agents[i].num_price - 1)
            oppo_price_count = np.sum(agents[i].q_nums, axis=0)
            total_sum = np.sum(oppo_price_count)
            s = oppo_price_count / total_sum
            agents[i].con_feed(
                [s, prices[i], int(round(prices[(i + 1) % env.num_firms], decimals) * agents[i].num_price - 1),
                 rewards[i]])
            agents[i].update(prices[i], prices[(i + 1) % env.num_firms], rewards[i])
        else:
            # print(i, (i+1)%env.num_firms)
            theprice_index = int(prices[i]*agents[i].num_price - 1)
            # print(theprice_index)
            price_count = np.sum(agents[i].q_nums[theprice_index])
            oppo_price_count = np.sum(agents[i].q_nums[:,theprice_index])
            q_freq = agents[i].q_nums[theprice_index] / price_count
            oppo_q_freq = agents[i].q_nums[:,theprice_index] / oppo_price_count
            s = np.concatenate((agents[i].q_value[theprice_index], q_freq))
            oppo_s = np.concatenate((agents[i].q_value[:,theprice_index], oppo_q_freq))
            if adj_smooth:
                agents[i].adj_feed([s,oppo_s, prices[i], int(prices[(i+1)%env.num_firms]*agents[i].num_price - 1), rewards[i]])
            else:
                agents[i].feed([s, prices[i], int(prices[(i + 1) % env.num_firms] * agents[i].num_price - 1), rewards[i]])
            # print(s, prices[i], int(prices[(i+1)%env.num_firms]*agents[i].num_price - 1), rewards[i])
            agents[i].update(prices[i], prices[(i+1)%env.num_firms], rewards[i])

# print(agents[0].q_value)

plot(price_record, time, file_path)
datasave(price_record, time, file_path)

for i in range(env.num_firms):

    if continueous_price:
        model_name_prob = 'model_cifar_prob_' + str(i) + '.pt'
        torch.save(agents[i].prob_policy.state_dict(), model_name_prob)
        model_name_price = 'model_cifar_price_' + str(i) + '.pt'
        torch.save(agents[i].price_policy.state_dict(), model_name_price)
    else:
        model_name = 'model_cifar_' + str(i) + '.pt'
        torch.save(agents[i].q_policy.state_dict(), model_name)
    if adj_smooth:
        print(agents[i].adj_neural_estimate())
    elif continueous_price:
        print(agents[i].continueous_neural_estimate())
    else:
        print(agents[i].neural_estimate())
