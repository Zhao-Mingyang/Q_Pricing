from Q_agent import Agent
from env import Env
import numpy as np
from utils import *
file_path = 'C:/Users/13862/Desktop/Pricing/experiments/'
total = 100
final_value = [[0 for _ in range(total)] for _ in range(2)]
for time in range(total):
    env = Env()
    agents = [Agent() for i in range(env.num_firms)]
    print(agents[0].q_value)
    T = 1000000
    price_record = [[0 for i in range(T)] for _ in range(env.num_firms)]
    for t in range(T):
        prices = [0 for i in range(env.num_firms)]
        for i in range(env.num_firms):
            prices[i] = agents[i].eplison_greedy()
            price_record[i][t] = prices[i]
        rewards = env.reward(prices)
        # print(prices)
        for i in range(env.num_firms):
            # print(i, (i+1)%env.num_firms)
            agents[i].update(prices[i], prices[(i+1)%env.num_firms], rewards[i])
        # break
    # print(agents[0].q_value)

    datasave(price_record, time, file_path)
    plot(price_record, time, file_path)
    for i in range(env.num_firms):
        final_value[i][time] = agents[i].maxexpected_policy()
        print(agents[i].maxexpected_policy())
finaldatasave(final_value, file_path)