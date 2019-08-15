import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
import itertools
from scipy.special import comb
import time


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

        # , header=None)  # , index_col=0)
        self.data = pd.read_csv(path)  # , header=None)  # index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.parameters = torch.ones(
            [self.dim] * self.order, dtype=torch.float32)
        # for x in self.data.columns:
        #     index = 0
        #     self.cats.append(np.unique(self.data[x]))
        #     self.data[x].replace(
        #         {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
        #     index += 1

    def funcvalue(self, x):
        # density

        x = torch.tensor(x, dtype=torch.float32)
        return self.normalize() * torch.exp(vec_tens_prod(x, self.parameters))

    def normalize(self):
        # normalization of prob density

        li = list(itertools.product([0, 1], repeat=self.dim))
        s = 0

        for x in li:
            s += torch.exp(vec_tens_prod(torch.tensor(x,
                                                      dtype=torch.float32), self.parameters))

        return 1/s

    def modeltest(self):
        li = list(itertools.product([0, 1], repeat=self.dim))

        for x in li:
            print(x, ": ", round(float(self.funcvalue(x)), 5))

        print(self.data.groupby(self.data.columns.tolist(), as_index=False).size())

    # def pseudoLH_correct(self, Q):
    #     # Correct seriell version of pseudoLH function, with torch
    #     s = 0
    #     r0, r1, r2, r3 = 0, 0, 0, 0
    #     for x in torch.tensor(np.array(self.data)):
    #
    #         for i in range(0, len(Q)):
    #             slices = cut_rth_slice(Q, i)
    #             slice1, slice2, slice3 = slices[0], slices[1], slices[2]
    #             z1, n1 = 0, 0
    #             z2, n2 = 0, 0
    #             z3, n3 = slice1 * x[i], slice1
    #
    #             for j in range(0, len(x)):
    #                 z2 += slice2[j] * x[i] * x[j]
    #                 n2 += slice2[j] * x[j]
    #
    #             for j in range(0, len(x)):
    #                 for k in range(0, len(x)):
    #                     z1 += slice3[j][k] * x[i] * x[j] * x[k]
    #                     n1 += slice3[j][k] * x[j] * x[k]
    #
    #             # print(i, ": ", 3 * z1 - 3 * z2 + z3)
    #             ergeb = 3 * z1 - 3 * z2 + z3 - \
    #                 torch.log(1 + torch.exp(3 * n1 - 3 * n2 + n3))
    #             s += ergeb
    #
    #     return -s/len(self.data)

    def pseudolh_2D(self, Q):
        s = 0
        for x in torch.tensor(np.array(self.data), dtype=torch.float32):
            for r in range(0, 1):
                a = 0
                for j in range(0, 1):
                    if (r == j):
                        a += Q[r][r]
                    else:
                        a += Q[r][j] * x[j]

                for j in range(0, 1):
                    s += Q[r][r] * x[r] * x[j] - torch.log(1 + torch.exp(a))

        return -s/len(self.data)

    def pseudoLH(self, Q):
        # Pseudo LH function vectorized, with torch

        data = torch.tensor(np.array(self.data), dtype=torch.float32)
        n, d = data.shape
        s = 0
        ones = torch.ones(n)
        zeros = torch.zeros((n, 1))
        for r in range(0, len(Q)):
            W = 0
            slices = []
            sli = Q
            for k in range(0, Q.dim()):
                sli = torch.unbind(sli, dim=0)[r]
                slices.append(sli)

            slice2d, slice1d, slice0d = slices[0], slices[1], slices[2]

            rth_col = data[:, r]

            W += slice0d * ones
            W -= 3 * torch.matmul(data, slice1d)
            W += 3 * \
                torch.sum(torch.mul(torch.matmul(data, slice2d), data), dim=1)

            W = torch.mul(
                W, rth_col) - torch.logsumexp(torch.cat((zeros, W.reshape((n, 1))), dim=1), dim=1)
            W = torch.sum(W)
            s -= W

        return s/n

    def regularization_term(self, Q):
        # regularization of objective function with nuclear norm
        return nuclear_norm_tens(Q)

    def obj_func(self, Q):
        # objective function

        #a = self.pseudolh_2D(Q)
        a = self.pseudoLH(Q)
        #b = 0.003 * torch.sum(torch.abs(Q))
        #c = 0.1 * self.regularization_term(Q)

        # print("PLH: ", a)
        # print("np.sum np.abs(S): ", b)
        # print("Nuclearnorm(L): ", c)

        return a  # + c

    def obj_referenz_sol(self):
        # Reference solution of Matlab (Frank) to test own pseudoLH
        S = torch.tensor([[[-0.25846, 3.7425e-11, 0.16237, 0.71328], [3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [0.71328, -6.938e-13, -1.3673e-12, 0.84142]], [[3.7425e-11, 0.21068, -8.5876e-13, -6.938e-13], [0.21068, -0.17064, 0.61465, 1.7338e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12]], [
            [0.16237, -8.5876e-13, 0.40328, -1.3673e-12], [-8.5876e-13, 0.61465, 0.61484, -8.5998e-13], [0.40328, 0.61484, -3.6873e-14, 2.0454e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12]], [[0.71328, -6.938e-13, -1.3673e-12, 0.84142], [-6.938e-13, 1.7338e-12, -8.5998e-13, 1.2212e-12], [-1.3673e-12, -8.5998e-13, 2.0454e-12, 1.2685e-12], [0.84142, 1.2212e-12, 1.2685e-12, -1.4665]]], dtype=torch.float32)
        L = torch.tensor([[[-1.6353, 0.68948, 0.68493, 0.66206], [0.68948, 0.5492, -0.56446, -0.48878], [0.68493, -0.56446, 0.4681, -0.61089], [0.66206, -0.48878, -0.61089, 0.56415]], [[0.68948, 0.5492, -0.56446, -0.48878], [0.5492, -1.2094, 0.5237, 0.43629], [-0.56446, 0.5237, 0.48314, -0.36059], [-0.48878, 0.43629, -0.36059, 0.4264]], [
            [0.68493, -0.56446, 0.4681, -0.61089], [-0.56446, 0.5237, 0.48314, -0.36059], [0.4681, 0.48314, -0.67632, 0.41368], [-0.61089, -0.36059, 0.41368, 0.42124]], [[0.66206, -0.48878, -0.61089, 0.56415], [-0.48878, 0.43629, -0.36059, 0.4264], [-0.61089, -0.36059, 0.41368, 0.42124], [0.56415, 0.4264, 0.42124, -1.0905]]], dtype=torch.float32)

        # Q = torch.tensor([[[0.3149, 0.4252, 0.3869, 0.2432],
        #                    [0.4252, 0.3411, 0.6358, 0.4482],
        #                    [0.3869, 0.6358, 0.1546, 0.3821],
        #                    [0.2432, 0.4482, 0.3821, 0.6282]],
        #
        #                   [[0.4252, 0.3411, 0.6358, 0.4482],
        #                    [0.3411, 0.1269, 0.3575, 0.4499],
        #                    [0.6358, 0.3575, 0.6116, 0.4635],
        #                    [0.4482, 0.4499, 0.4635, 0.7458]],
        #
        #                   [[0.3869, 0.6358, 0.1546, 0.3821],
        #                    [0.6358, 0.3575, 0.6116, 0.4635],
        #                    [0.1546, 0.6116, 0.8924, 0.3595],
        #                    [0.3821, 0.4635, 0.3595, 0.3807]],
        #
        #                   [[0.2432, 0.4482, 0.3821, 0.6282],
        #                    [0.4482, 0.4499, 0.4635, 0.7458],
        #                    [0.3821, 0.4635, 0.3595, 0.3807],
        #                    [0.6282, 0.7458, 0.3807, 0.8537]]], dtype=torch.float32)

        Q = S + L

        #start = time.time()
        # for i in range(0, 2000):
        a = self.pseudoLH(Q)
        b = 0.003 * torch.sum(torch.abs(S))
        c = 0.01 * self.regularization_term(L)
        d = a + b + c
        #ende = time.time()
        # print('{:5.3f}s'.format(ende-start))

        # print("PLH: ", a)
        # print("np.sum np.abs(S): ", b)
        # print("Nuclearnorm(L): ", c)

        return d, Q

    def testoptimize(self, iter, param=0.01):
        # Solution for objective function based on torch with gradient

        # Q = torch.tensor([[[-0.6191,  1.0152,  1.2865,  0.9622],
        #          [1.0456, -0.1037,  0.2706,  0.1418],
        #          [1.3233,  0.1646, -0.0330,  0.3135],
        #          [1.6803,  0.2282,  0.3145,  0.2792]],
        #
        #         [[-0.0884,  1.0149,  0.6350,  0.0939],
        #          [1.0840, -0.3488,  1.6547,  0.6056],
        #          [0.5810,  1.6409,  0.2639,  0.4770],
        #          [0.2031,  0.4148,  0.2138, -0.5627]],
        #
        #         [[-0.1948,  0.7164,  1.2132,  0.0133],
        #          [0.7671,  0.0254,  1.6966,  0.3721],
        #          [1.1388,  1.4386,  0.2132,  0.7427],
        #          [0.2868,  0.2215,  0.2939, -0.4157]],
        #
        #         [[0.4691,  0.3072,  0.2296,  1.6444],
        #          [0.3201, -0.5200, -0.0240,  0.1243],
        #          [0.2452, -0.0068, -0.5162,  0.1170],
        #          [1.1939,  0.5564,  0.6714, -1.1124]]], requires_grad=True)

        # Solution: tensor(0.3333, grad_fn=<AddBackward0>)

        Q = torch.zeros([self.dim] * self.order,
                        dtype=torch.float32, requires_grad=True)

        # Q = torch.tensor([[[-6., -6., -6.],
        #                    [-6., -6., -6.],
        #                    [-6., -6., -6.]],
        #
        #                   [[-6., -6., -6.],
        #                    [-6., -6., -6.],
        #                    [-6., -6., -6.]],
        #
        #                   [[-6., -6., -6.],
        #                    [-6., -6., -6.],
        #                    [-6., -6., -6.]]], dtype=torch.float32, requires_grad=True)

        #Q = make_tens_symm(Q)

        # Q = torch.tensor([[[-1., -1., -1.],
        #                    [-1., -1., -1.],
        #                    [-1., -1., -1.]],
        #
        #                   [[-1., -1., -1.],
        #                    [-1., 1., -1.],
        #                    [-1., -1., -1.]],
        #
        #                   [[-1., -1., -1.],
        #                    [-1., -1., -1.],
        #                    [-1., -1., -1.]]], dtype=torch.float32, requires_grad=True)

        # Q = Q - torch.ones([self.dim] * self.order,
        #                    dtype=torch.float32, requires_grad=True)
        print(Q)
        
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
            # Q = make_tens_symm(Q)

        print(Q, s)

        return Q

        def modeleselect_ADMM(self, iter):

            Q = torch.zeros([self.dim] * self.order,
                            dtype=torch.float32, requires_grad=True)

            return Q

    def f_proximal_opt_sol(self, L, M, Phi, mu, iterTO):

        L = torch.tensor(L, dtype=torch.float32, requires_grad=True)

        s = self.pseudoLH(L) + (2 * mu)**(-1) * \
            torch.frobenius_norm(L - M + mu * Phi)

        optimizer = torch.optim.Adam([L], lr=0.0001)

        s = 0
        for i in range(iterTO):
            optimizer.zero_grad()

            s = self.pseudoLH(L) + (2 * mu)**(-1) * \
                torch.frobenius_norm(L - M + mu * Phi)

            s.sum().backward(retain_graph=True)
            optimizer.step()

        return L

    def g_proximal_opt_sol(self, L, M, Phi, mu, iterTO):

        M = torch.tensor(M, dtype=torch.float32, requires_grad=True)

        s = nuclear_norm_tens(M) + (2 * mu)**(-1) * \
            torch.frobenius_norm(L - M + mu * Phi)

        optimizer = torch.optim.Adam([M], lr=0.0001)

        s = 0
        for i in range(iterTO):
            optimizer.zero_grad()

            s = nuclear_norm_tens(M) + (2 * mu)**(-1) * \
                torch.frobenius_norm(L - M + mu * Phi)

            s.sum().backward(retain_graph=True)
            optimizer.step()

        return M

    def modeleselect_ADMM(self, iterADMM, iterTO):
        # Modelselection for model with ADMM algorithm
        Y = torch.zeros([self.dim] * self.order,
                        dtype=torch.float32, requires_grad=True)

        # L = torch.eye([self.dim] * self.order,
        #                 dtype=torch.float32, requires_grad=True)
        L = Y
        M = L
        mu = 0.1

        for i in range(0, iterADMM):
            print(i)
            L = self.f_proximal_opt_sol(L, M, Y, mu, iterTO)
            #L = make_tens_symm(L)

            M = self.g_proximal_opt_sol(L, M, Y, mu, iterTO)
            #M = make_tens_symm(M)

            Y = Y + mu**(-1) * (L - M)

        return L, M, Y
