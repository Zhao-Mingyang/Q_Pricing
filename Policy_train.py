from Q_agent import Agent
from env import Env
import matplotlib.pyplot as plt
import numpy as np
from utils import *

env = Env()
agents = [Agent() for i in range(env.num_firms)]
print(agents[0].q_value)
T = 100000
price_record = [[0 for i in range(T)] for _ in range(env.num_firms)]
for t in range(T):
    prices = [0 for i in range(env.num_firms)]
    for i in range(env.num_firms):
        prices[i] = agents[i].eplison_greedy()
        price_record[i][t] = prices[i]
    rewards = env.reward(prices)
    print(prices)
    for i in range(env.num_firms):
        print(i, (i+1)%env.num_firms)
        agents[i].update(prices[i], prices[(i+1)%env.num_firms], rewards[i])

# print(agents[0].q_value)

plot(price_record)
for i in range(env.num_firms):
    print(agents[i].maxmean_policy())