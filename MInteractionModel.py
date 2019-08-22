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

        self.data_all = torch.tensor(np.array(self.data), dtype=torch.float32)

        self.li_comb = list(itertools.product([0, 1], repeat=self.dim))
        self.data_comb = torch.tensor(
            np.array(self.li_comb), dtype=torch.float32)

        self.multiplicities = []
        for x in self.data_comb:
            self.multiplicities.append((self.data_all == x).all(axis=1).sum())

        self.Q = torch.ones([self.dim] * self.order, dtype=torch.float32)

    def funcvalue(self, x, normalize=True):

        x = torch.tensor(x, dtype=torch.float32)
        if normalize:
            return torch.exp(vec_tens_prod(x, self.Q)) * self.normalize()
        else:
            return torch.exp(vec_tens_prod(x, self.Q))

    def normalize(self):
        f = 0

        for x in self.li_comb:
            f += torch.exp(vec_tens_prod(torch.tensor(x,
                                                      dtype=torch.float32), self.Q))

        return 1/f

    def pseudoLH(self, Q):

        data = self.data_comb
        multiplicities = torch.tensor(
            np.array(self.multiplicities), dtype=torch.float32)

        n, d = data.shape
        s = 0
        ones = torch.ones(n)
        zeros = torch.zeros((n, 1))
        W = 0

        for r in range(0, len(Q)):
            slices = cut_rth_slice(Q, r)
            rth_col = data[:, r]
            data_denom = torch.tensor(
                np.array(self.li_comb), dtype=torch.float32)
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

    def new_pseudoLH(self):

        s = 0
        for x in self.li_comb:
            f = round(float(self.funcvalue(x, normalize)), 5)
            frequ = (self.data == x).all(axis=1).sum()
            #print(x, " --- ", frequ, " --- p(x) = ", f)
            s += np.abs(f - frequ/self.len)

        return s

    def modeltest(self, normalize=True):

        print("x --- Frequiencies (total: ", self.len,
              ") --- Prediction of the model")

        s = 0
        for x in self.li_comb:
            f = round(float(self.funcvalue(x, normalize)), 5)
            frequ = (self.data == x).all(axis=1).sum()
            print(x, " --- ", frequ, " --- p(x) = ", f)
            s += np.abs(f - frequ/self.len)

        print("Deviation: ", s/len(self.li_comb))

    def obj_func(self, Q, S=torch.zeros(1), L=torch.zeros(1), a=0, b=0):

        # + a * torch.sum(torch.abs(S)) + b * nuclear_norm_tens(L)
        return self.pseudoLH(Q)

    def torch_optimize(self, iter, seedpoint=1, param=0.01, optim_alg="ASGD"):

        if torch.is_tensor(seedpoint):
            Q = seedpoint
        else:
            Q = torch.zeros([self.dim] * self.order,
                            dtype=torch.float32, requires_grad=True)

        s = self.obj_func(Q)  # , S=Q, a=0.05)
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
        for i in range(iter):
            optimizer.zero_grad()

            s = self.obj_func(Q)  # , S=Q, a=0.05)

            s.sum().backward(retain_graph=True)
            optimizer.step()

            if (i % 1000 == 0):
                print(i)
                if (s == s_old):
                    print("equal.")
                    break
                s_old = s

        #print(Q, "\n", float(s))

        return Q

    def torch_optimize_with_symmetry(self, outer_iter, iter, seedpoint=1, param=0.01, dec_rate=1, optim_alg="ASGD"):

        Q = torch.zeros([self.dim] * self.order,
                        dtype=torch.float32, requires_grad=True)

        best_fval = np.inf
        current_tens = Q
        param_start = param
        for i in range(outer_iter):
            print("Outer Index: ", i, " LR: ",
                  param, " Inner_max_iter: ", iter)
            Q = self.torch_optimize(
                iter=iter, seedpoint=current_tens, param=param, optim_alg=optim_alg)

            current_tens = make_tens_super_symm(Q.clone())
            current_fval = float(self.pseudoLH(current_tens))
            print(current_fval - best_fval)
            if (current_fval < best_fval):
                best_fval = current_fval
                self.Q = current_tens.clone()
                print(self.Q)
                print(best_fval)
            else:
                param -= dec_rate
                if param < 0.00000001:
                    param = param_start

        return self.Q

    def matlab_referenz_sol(self):

        self.add_dataset("../referenz_data.csv")
        self.S = torch.tensor([[[-0.25846, 3.7425e-11, 0.16237, 0.71328], [3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [0.71328, -6.938e-13, -1.3673e-12, 0.84142]], [[3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.21068, -0.17064, 0.61465, 1.7338e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12]], [
            [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [0.40328, 0.61484, -3.6873e-14, 2.0454e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12]], [[0.71328, -6.938e-13, -1.3673e-12, 0.84142], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12], [0.84142, 1.2212e-12, 1.2685e-12, -1.4665]]], dtype=torch.float32)
        self.L = torch.tensor([[[-1.6353, 0.68948, 0.68493, 0.66206], [0.68948, 0.5492, -0.56446, -0.48878], [0.68493, -0.56446, 0.4681, -0.61089], [0.66206, -0.48878, -0.61089, 0.56415]], [[0.68948, 0.5492, -0.56446, -0.48878], [0.5492, -1.2094, 0.5237, 0.43629], [-0.56446, 0.5237, 0.48314, -0.36059], [-0.48878, 0.43629, -0.36059, 0.4264]], [
            [0.68493, -0.56446, 0.4681, -0.61089], [-0.56446, 0.5237, 0.48314, -0.36059], [0.4681, 0.48314, -0.67632, 0.41368], [-0.61089, -0.36059, 0.41368, 0.42124]], [[0.66206, -0.48878, -0.61089, 0.56415], [-0.48878, 0.43629, -0.36059, 0.4264], [-0.61089, -0.36059, 0.41368, 0.42124], [0.56415, 0.4264, 0.42124, -1.0905]]], dtype=torch.float32)

        self.Q = self.S + self.L
