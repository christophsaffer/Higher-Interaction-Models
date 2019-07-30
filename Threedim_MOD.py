import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
import itertools
from scipy.special import comb

from tools import *


class Threedim_MOD:

    def __init__(self):

        self.order = 3
        self.data = 0
        self.dim = 0
        self.len = 0
        self.cats = []
        # self.parameters = torch.tensor()

    def add_dataset(self, path):

        self.data = pd.read_csv(path)  # , index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.parameters = torch.ones(
            [self.dim] * self.order, dtype=torch.float32)
        for x in self.data.columns:
            index = 0
            self.cats.append(np.unique(self.data[x]))
            self.data[x].replace(
                {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
            index += 1

    def funcvalue(self, x):

        x = torch.tensor(x, dtype=torch.float32)
        return self.normalize() * np.exp(vec_tens_prod(x, self.parameters))

    def normalize(self):

        li = list(itertools.product([0, 1], repeat=self.dim))
        s = 0

        for x in li:
            s += np.exp(vec_tens_prod(torch.tensor(x,
                                                   dtype=torch.float32), self.parameters))

        return 1/s

    def node_conditional(self, Q, x, r):

        slices = cut_rth_slice(Q, r)

        ind = 0
        prod = 1
        for slice in slices:
            # print(slice)
            # print(x)
            # a = vec_tens_prod(x, slice)
            # print(a)
            prod *= torch.exp((-1)**(ind) * comb(self.dim, ind)
                              * vec_tens_prod(x, slice))

            ind += 1

        return torch.log(prod)

    def normalize_node_c(self, Q, x, r):

        slices = cut_rth_slice(Q, r)

        x[r] = 1

        ind = 0
        prod = 1
        for slice in slices:
            prod *= torch.exp((-1)**(ind) * comb(self.dim, ind)
                              * vec_tens_prod(x, slice))
            ind += 1

        return torch.log(1 + prod)

    def pseudoLH(self, Q):
        s = 0

        # Q = Q.reshape([self.dim] * self.order)
        # Q = torch.tensor(Q, dtype=torch.float32)

        for x in torch.tensor(np.array(self.data)):
            for i in range(0, Q.dim()):
                s += self.node_conditional(Q, x, i) - \
                    self.normalize_node_c(Q, x, i)

        return -s

    def modelselect(self):

        sol = minimize(self.pseudoLH, self.parameters, method='BFGS')
        Q = sol['x'].reshape([self.dim] * self.order)
        self.parameters = torch.tensor(Q, dtype=torch.float32)

        return sol

    def testoptimize(self, iter):

        Q = torch.zeros([self.dim] * self.order,
                        dtype=torch.float32, requires_grad=True)

        # with 1500 Iter #funval: -211.7291
        # Q = torch.tensor([[[1.4767, -1.5002, -0.3283],
        #                    [1.3241, -1.3061,  1.4472],
        #                    [-1.0795,  1.2505, -0.7692]],
        #
        #                   [[0.0000,  0.0000,  1.4522],
        #                    [-1.5002,  1.4517, -0.3284],
        #                    [0.0000,  0.0000, -1.5913]],
        #
        #                   [[0.0000,  0.0000, -1.5002],
        #                    [1.4693, -1.6198, -1.5188],
        #                    [0.0000,  0.0000,  1.4522]]], dtype=torch.float32, requires_grad=True)

        # Q = torch.tensor([[[0.0207, -0.0809, -0.0226,  0.0712],
        #                    [0.0653, -0.0226,  0.0742,  0.0743],
        #                    [-0.0844,  0.0748,  0.0378,  0.0669],
        #                    [0.0727,  0.0626,  0.0680,  0.0731]],
        #
        #                   [[0.0000,  0.0000,  0.0743,  0.0000],
        #                    [-0.0816, -0.0176,  0.0722, -0.0809],
        #                    [0.0000,  0.0000, -0.0974,  0.0000],
        #                    [0.0000,  0.0000,  0.0681,  0.0000]],
        #
        #                   [[0.0000,  0.0000, -0.0820,  0.0000],
        #                    [0.0738,  0.0648, -0.0936,  0.0639],
        #                    [0.0000,  0.0000, -0.0158,  0.0000],
        #                    [0.0000,  0.0000, -0.0815,  0.0000]],
        #
        #                   [[0.0000,  0.0000,  0.0681,  0.0000],
        #                    [0.0747, -0.0839,  0.0678,  0.0747],
        #                    [0.0000,  0.0000, -0.0861,  0.0000],
        #                    [0.0000,  0.0000,  0.0681,  0.0000]]],  dtype=torch.float32, requires_grad=True)

        # Q = torch.tensor([[[-0.25846, 3.7425e-11, 0.16237, 0.71328], [3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [0.71328, -6.938e-13, -1.3673e-12, 0.84142]], [[3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.21068, -0.17064, 0.61465, 1.7338e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12]],
        # [[0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [0.40328, 0.61484, -3.6873e-14, 2.0454e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12]], [[0.71328, -6.938e-13, -1.3673e-12, 0.84142], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12], [0.84142, 1.2212e-12, 1.2685e-12, -1.4665]]], dtype=torch.float32, requires_grad=True)

        s = self.pseudoLH(Q)

        optimizer = torch.optim.Adam([Q], lr=0.0001)

        s = 0
        for i in range(iter):
            print(i)
            optimizer.zero_grad()

            s = self.pseudoLH(Q)

            s.sum().backward(retain_graph=True)
            optimizer.step()
            #Q = make_tens_symm(Q)

        print(Q, s)

        return Q

        def modeleselect_ADMM(self, iter):

            Q = torch.zeros([self.dim] * self.order,
                            dtype=torch.float32, requires_grad=True)

            return Q
