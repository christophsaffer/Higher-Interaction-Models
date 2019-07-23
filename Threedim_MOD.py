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
