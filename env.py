import numba
from numba import jit
import random
import numpy as np

class Env(object):
    def __init__(self):
        self.num_firms = 2

    def reward(self, prices):
        d0 = max(0.48 - 0.9 * prices[0] + 0.6 * prices[1],0)
        d1 = max(0.48 - 0.9 * prices[1] + 0.6 * prices[0],0)
        return [prices[0] * d0, prices[1] * d1]
