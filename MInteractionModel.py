import numpy as np
import torch
import pandas as pd
import itertools
from scipy.special import comb

from tools import *


class MInteractionModel:

    def __init__(self, order):

        self.order = order
        self.Q = 0

    def add_dataset(self, path):
        self.data = pd.read_csv(path)  # , header=None)  # index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.Q = torch.ones([self.dim] * self.order, dtype=torch.float32)

    def funcvalue(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        return self.normalize() * torch.exp(vec_tens_prod(x, self.Q))

    def normalize(self):

        li = list(itertools.product([0, 1], repeat=self.dim))
        f = 0

        for x in li:
            f += torch.exp(vec_tens_prod(torch.tensor(x,
                                                      dtype=torch.float32), self.Q))

        return 1/f

    def pseudoLH(self, Q):

        data = torch.tensor(np.array(self.data), dtype=torch.float32)
        n, d = data.shape
        s = 0
        ones = torch.ones(n)
        zeros = torch.zeros((n, 1))

        for r in range(0, len(Q)):
            W = 0

            slices = cut_rth_slice(Q, r)

            rth_col = data[:, r]

            if self.order == 2:
                data_denom = torch.tensor(
                    np.array(self.data), dtype=torch.float32)
                data_denom[:, r] = ones
                W = - slices[0] * ones
                W1 = W + 2 * torch.matmul(data, slices[1])
                W2 = W + 2 * torch.matmul(data_denom, slices[1])
            else:
                W += slices[0] * ones
                W -= 3 * torch.matmul(data, slices[1])
                W += 3 * \
                    torch.sum(torch.mul(torch.matmul(
                        data, slices[2]), data), dim=1)
                W1, W2 = W, W

            W = torch.mul(
                W1, rth_col) - torch.logsumexp(torch.cat((zeros, W2.reshape((n, 1))), dim=1), dim=1)
            s -= torch.sum(W)

        return s/n

    def modeltest(self):
        li = list(itertools.product([0, 1], repeat=self.dim))

        for x in li:
            print(x, ": ", round(float(self.funcvalue(x)), 5))

        print(self.data.groupby(self.data.columns.tolist(), as_index=False).size())
