import numpy as np
import pandas as pd
import itertools


class Ising_MOD:

    def __init__(self):
        self.name = "test"

    def add_dataset(self, path):

        self.data = pd.read_csv(path)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.parameters = []
        for x in self.data.columns:
            index = 0
            self.cats.append(np.unique(self.data[x]))
            self.data[x].replace(
                {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
            index += 1

    def slicefunc(self, Q, x, r):
        s = 0
        for i in range(0, self.dim):
            s += 2 * Q[r, i] * x[r] * x[i]

        return s

    def normalizfunc(self, Q, x, r):
        s = 0
        for i in range(0, self.dim):
            s += 2 * Q[r, i] * x[i]

        return np.log(1 + np.exp(s))

    def pseudoLH(self, Q):
        s = 0

        for x in np.array(self.data):
            for i in range(0, self.dim):
                s += self.slicefunc(Q, x, i) - self.normalizfunc(Q, x, i)

        return -s

    # def modelselection(self):


if __name__ == '__main__':

    testobj = Ising_MOD()
    testobj.add_dataset("../cgmodelselection/unittest_data/py_D_s12d3l0.csv")
