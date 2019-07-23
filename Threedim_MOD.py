import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
import itertools

from tools import *


class Threedim_MOD:

    def __init__(self):

        self.order = 3
        self.data = 0
        self.dim = 0
        self.len = 0
        self.cats = []
        self.parameters = torch.tensor()

    def add_dataset(self, path):

        self.data = pd.read_csv(path, index_col=0)
        self.dim = len(self.data.columns)
        self.len = len(self.data)
        self.cats = []
        self.parameters = torch.ones([self.dim] * self.order)
        for x in self.data.columns:
            index = 0
            self.cats.append(np.unique(self.data[x]))
            self.data[x].replace(
                {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
            index += 1

    def funcvalue(self, x):

        x = torch.tensor(x)
        return self.normalize() * np.exp(tens_vec_prod(x, self.parameters))

    def normalize(self):

        li = list(itertools.product([0, 1], repeat=self.dim))
        s = 0

        for x in li:
            s += np.exp(tens_vec_prod(x, self.parameters))

        return 1/s

    # def slicefunc(self):
    # def pseudoLH(self):
    # def modelselect(self):
