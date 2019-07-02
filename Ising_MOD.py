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
        # index = 0
        for x in self.data.columns:
            index = 0
            self.cats.append(np.unique(self.data[x]))
            self.data[x].replace(
                {self.cats[index][0]: 0, self.cats[index][1]: 1}, inplace=True)
            index += 1

    # def pseudoLH()

    # def modelselection(self):


if __name__ == '__main__':

    testobj = Ising_MOD()
    testobj.add_dataset("../cgmodelselection/unittest_data/py_D_s12d3l0.csv")
