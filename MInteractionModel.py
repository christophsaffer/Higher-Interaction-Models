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
        W = 0

        for r in range(0, len(Q)):
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
                data_denom = torch.tensor(
                    np.array(self.data), dtype=torch.float32)
                data_denom[:, r] = ones
                # W = slices[0] * ones
                # W -= 3 * torch.matmul(data, slices[1])
                # W += 3 * \
                #     torch.sum(torch.mul(torch.matmul(
                #         data, slices[2]), data), dim=1)
                # W1, W2 = W, W

                W = slices[0] * ones
                W1 = W - 3 * torch.matmul(data, slices[1])
                W1 += 3 * \
                    torch.sum(torch.mul(torch.matmul(
                        data, slices[2]), data), dim=1)
                W2 = W - 3 * torch.matmul(data_denom, slices[1])
                W2 += 3 * \
                    torch.sum(torch.mul(torch.matmul(
                        data_denom, slices[2]), data_denom), dim=1)

            W = torch.mul(
                W1, rth_col) - torch.logsumexp(torch.cat((zeros, W2.reshape((n, 1))), dim=1), dim=1)
            s -= torch.sum(W)

        return s/n

    def modeltest(self, count_data=True, verbose=True):
        li = list(itertools.product([0, 1], repeat=self.dim))
        results = []
        if verbose:
            print("\nPrediction of the model:")
        for x in li:
            f = round(float(self.funcvalue(x)), 5)
            results.append(f)
            if verbose:
                print("p(", x, ") = ", f)

        if (count_data):
            if verbose:
                print("\nFrequiencies in the dataset:")
                print(self.data.groupby(
                    self.data.columns.tolist(), as_index=False).size())

        return results

    def obj_func(self, Q, S=torch.zeros(1), L=torch.zeros(1), a=0, b=0):

        return self.pseudoLH(Q) + a * torch.sum(torch.abs(S)) + b * nuclear_norm_tens(L)

    def torch_optimize(self, iter, param=0.01):

        Q = torch.zeros([self.dim] * self.order,
                        dtype=torch.float32, requires_grad=True)

        s = self.obj_func(Q)

        optimizer = torch.optim.ASGD([Q], lr=param)

        s = 0
        for i in range(iter):
            if (i % 100 == 0):
                print(i)
            optimizer.zero_grad()

            s = self.obj_func(Q)

            s.sum().backward(retain_graph=True)
            optimizer.step()
        print(Q, s)

        return Q

    def matlab_referenz_sol(self):

        self.add_dataset("../referenz_data.csv")
        self.S = torch.tensor([[[-0.25846, 3.7425e-11, 0.16237, 0.71328], [3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [0.71328, -6.938e-13, -1.3673e-12, 0.84142]], [[3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.21068, -0.17064, 0.61465, 1.7338e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12]], [
            [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [0.40328, 0.61484, -3.6873e-14, 2.0454e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12]], [[0.71328, -6.938e-13, -1.3673e-12, 0.84142], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12], [0.84142, 1.2212e-12, 1.2685e-12, -1.4665]]], dtype=torch.float32)
        self.L = torch.tensor([[[-1.6353, 0.68948, 0.68493, 0.66206], [0.68948, 0.5492, -0.56446, -0.48878], [0.68493, -0.56446, 0.4681, -0.61089], [0.66206, -0.48878, -0.61089, 0.56415]], [[0.68948, 0.5492, -0.56446, -0.48878], [0.5492, -1.2094, 0.5237, 0.43629], [-0.56446, 0.5237, 0.48314, -0.36059], [-0.48878, 0.43629, -0.36059, 0.4264]], [
            [0.68493, -0.56446, 0.4681, -0.61089], [-0.56446, 0.5237, 0.48314, -0.36059], [0.4681, 0.48314, -0.67632, 0.41368], [-0.61089, -0.36059, 0.41368, 0.42124]], [[0.66206, -0.48878, -0.61089, 0.56415], [-0.48878, 0.43629, -0.36059, 0.4264], [-0.61089, -0.36059, 0.41368, 0.42124], [0.56415, 0.4264, 0.42124, -1.0905]]], dtype=torch.float32)

        self.Q = self.S + self.L
