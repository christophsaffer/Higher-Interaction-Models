import numpy as np
import torch
import pandas as pd
import itertools
from scipy.special import comb

import tools


class MInteractionModel:

    def __init__(self, order, use_mult=False):

        self.order = order
        self.use_mult = use_mult
        self.Q = 0

    def add_dataset(self, path):
        self.data = pd.read_csv(path)  # , header=None)  # index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.data_all = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.li_comb = list(itertools.product([0, 1], repeat=self.dim))

        if (self.len > 2**self.dim) & (self.dim < 14) & (self.use_mult):
            print("Use pseudoLH with multiplicities as pseudoLH.")
            self.data_comb = torch.tensor(
                np.array(self.li_comb), dtype=torch.float32)

            self.multiplicities = []
            for x in self.data_comb:
                self.multiplicities.append(
                    (self.data_all == x).all(axis=1).sum())

            self.pseudoLH = self.pseudoLH_multiplicities

        else:
            print("Use pseudoLH with all data as pseudoLH.")
            self.pseudoLH = self.pseudoLH_all_data

        self.Q = torch.ones([self.dim] * self.order, dtype=torch.float32)
        self.symm_idx = tools.get_str_symm_idx_lst(self.Q)

    def funcvalue(self, x, normalize=True):

        x = torch.tensor(x, dtype=torch.float32)
        if normalize:
            return torch.exp(tools.vec_tens_prod(x, self.Q)) * self.normalize(self.Q)
        else:
            return torch.exp(tools.vec_tens_prod(x, self.Q))

    def normalize(self, Q):
        f = 0

        for x in self.li_comb:
            f += torch.exp(tools.vec_tens_prod(torch.tensor(x,
                                                            dtype=torch.float32), Q))

        return 1/f

    def pseudoLH_multiplicities(self, Q):

        data = self.data_comb
        multiplicities = torch.tensor(
            np.array(self.multiplicities), dtype=torch.float32)

        n, d = data.shape
        ones = torch.ones(n)
        zeros = torch.zeros((n, 1))
        s = 0
        for r in range(0, len(Q)):
            slices = tools.cut_rth_slice(Q, r)
            rth_col = data[:, r]
            data_denom = data.clone()
            data_denom[:, r] = ones

            if self.order == 2:
                W = - slices[0] * ones
                W1 = W + 2 * torch.matmul(data, slices[1])
                W2 = W + 2 * torch.matmul(data_denom, slices[1])
            else:
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
            s -= torch.sum(W * multiplicities)

        return s/self.len

    def pseudoLH_all_data(self, Q):

        data = self.data_all
        n, d = data.shape

        ones = torch.ones(n)
        zeros = torch.zeros((n, 1))
        s = 0
        for r in range(0, len(Q)):
            slices = tools.cut_rth_slice(Q, r)
            rth_col = data[:, r]
            data_denom = data.clone()
            data_denom[:, r] = ones

            if self.order == 2:
                W = - slices[0] * ones
                W1 = W + 2 * torch.matmul(data, slices[1])
                W2 = W + 2 * torch.matmul(data_denom, slices[1])
            else:
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

    def modeltest(self, normalize=True):

        print("x --- Frequiencies (total: ", self.len,
              ") --- Prediction of the model")

        s = 0
        for x in self.li_comb:
            f = round(float(self.funcvalue(x)), 5)
            frequ = (self.data == x).all(axis=1).sum()
            print(x, " --- ", frequ, " --- p(x) = ", f)
            s += np.abs(f - frequ/self.len)

        print("Deviation: ", s/len(self.li_comb))

    def obj_func(self, Q, S=torch.zeros(1), L=torch.zeros(1), a=0, b=0):

        Q = tools.make_tens_str_symm(Q.clone(), self.symm_idx)
        # Q = tools.make_tens_symm(Q)

        return self.pseudoLH(Q) + a * torch.sum(torch.abs(Q)) + b * tools.nuclear_norm_tens(Q)

    def torch_optimize(self, iter, seedpoint=1, param=0.01, optim_alg="ASGD", a=0, b=0):

        if torch.is_tensor(seedpoint):
            Q = seedpoint
        else:
            Q = torch.zeros([self.dim] * self.order,
                            dtype=torch.float32, requires_grad=True)

        s = self.obj_func(Q, a=a, b=b)
        if optim_alg == "ASGD":
            optimizer = torch.optim.ASGD([Q], lr=param)
        elif optim_alg == "Adam":
            optimizer = torch.optim.Adam([Q], lr=param)
        elif optim_alg == "SGD":
            optimizer = torch.optim.SGD(
                [Q], lr=param, momentum=0.5, nesterov=True)
        else:
            print("Algorithm ", optim_alg, " does not exist, using ASGD ...")
            optimizer = torch.optim.ASGD([Q], lr=param)

        s, s_old = 0, 0
        for i in range(1, iter):
            optimizer.zero_grad()

            s = self.obj_func(Q, a=a, b=b)

            s.sum().backward(retain_graph=True)
            optimizer.step()

            self.temp = Q.clone()
            if (i % 500 == 0):
                print("Iter =", i)
                print(Q, "\nFunVal:", float(s))
                if (s == s_old):
                    print("No more improvment.. stop.")
                    break
                s_old = s

        return Q
