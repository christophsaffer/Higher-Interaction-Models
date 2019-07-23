import numpy as np
import pandas as pd
from scipy.optimize import minimize
import itertools


class Ising_MOD:

    def __init__(self):
        self.name = "test"

    def add_dataset(self, path):

        self.data = pd.read_csv(path, index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.parameters = []
        # for x in self.data.columns:
        #     index = 0
        #     self.cats.append(np.unique(self.data[x]))
        #     self.data[x].replace(
        #         {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
        #     index += 1

    def slicefunc(self, Q, x, r):
        s = 0
        for i in range(0, self.dim):
            # s += 2 * Q[r, i] * x[r] * x[i]
            if i == r:
                s += 2 * Q[r * self.dim + i] * x[r] * x[i]
            else:
                s += Q[r * self.dim + i] * x[r] * x[i]

        return 0.5 * s

    def normalizfunc(self, Q, x, r):
        s = 0
        for i in range(0, self.dim):
            if i == r:
                s += 2 * Q[r * self.dim + i] * x[i]
            else:
                s += Q[r * self.dim + i] * x[i]

        return np.log(1 + np.exp(0.5 * s))

    def pseudoLH(self, Q):
        s = 0

        for x in np.array(self.data):
            for i in range(0, self.dim):
                s += self.slicefunc(Q, x, i) - self.normalizfunc(Q, x, i)

        return -s

    def modelselect(self):

        Q = np.zeros(self.dim * self.dim)
        sol = minimize(self.pseudoLH, Q, method='BFGS')
        self.parameters = sol['x']

        return sol

    def normalize(self):
        li = list(itertools.product([0, 1], repeat=self.dim))
        s = 0
        Q = self.parameters.reshape((self.dim, self.dim))

        for x in li:
            s += np.exp(0.5 * np.dot(np.dot(np.array(x), Q), np.array(x)))

        return 1/s

    def funcvalue(self, x):

        Q = self.parameters.reshape((self.dim, self.dim))

        return self.normalize() * np.exp(0.5 * np.dot(np.dot(np.array(x), Q), np.array(x)))


if __name__ == '__main__':

    testobj = Ising_MOD()
    testobj.add_dataset("../cgmodelselection/unittest_data/py_D_s12d3l0.csv")
