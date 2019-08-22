#!/usr/bin/python

import unittest
import numpy as np
import pandas as pd
import torch

import MInteractionModel
import tools


class TestTools(unittest.TestCase):

    def test_make_tens_symm_2order2(self):
        self.tens = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        symm_tens = torch.tensor([[1, 2.5], [2.5, 4]])
        self.assertTrue(torch.equal(tools.make_tens_symm(
            self.tens), symm_tens))

    def test_make_tens_symm_2order3(self):
        self.tens = torch.tensor(
            [[5, 1, 2], [1, 4, 3], [2, 3, 6]], dtype=torch.float32)
        symm_tens = torch.tensor(
            [[5, 1, 2], [1, 4, 3], [2, 3, 6]], dtype=torch.float32)
        self.assertTrue(torch.equal(
            tools.make_tens_symm(self.tens), symm_tens))

    def test_make_tens_symm_3order3(self):
        self.tens = torch.tensor(
            [[[5, 1, 2], [1, 4, 3], [2, 3, 10]], [[1, 4, 3], [4, 7, 9], [3, 9, 6]], [[2, 3, 10], [3, 9, 6], [10, 6, 8]]], dtype=torch.float32)
        symm_tens = torch.tensor(
            [[[5, 1, 2], [1, 4, 3], [2, 3, 10]], [[1, 4, 3], [4, 7, 9], [3, 9, 6]], [[2, 3, 10], [3, 9, 6], [10, 6, 8]]], dtype=torch.float32)
        self.assertTrue(torch.equal(
            tools.make_tens_symm(self.tens), symm_tens))

    def test_make_tens_symm_3order3_2(self):
        self.tens = torch.tensor([[[1, 3, 2], [4, 5, 4], [6, 3, 3]], [[5, 2, 1], [
                                 5, 1, 1], [7, 2, 3]], [[4, 2, 5], [1, 9, 6], [4, 3, 1]]], dtype=torch.float32)
        symm_tens = torch.tensor([[[1, 4, 4], [4, 4, 3], [4, 3, 4]], [[4, 4, 3], [
                                 4, 1, 4], [3, 4, 4]], [[4, 3, 4], [3, 4, 4], [4, 4, 1]]], dtype=torch.float32)
        self.assertTrue(torch.equal(
            tools.make_tens_symm(self.tens), symm_tens))

    def test_vec_tens_prod(self):
        self.tens = torch.tensor([[[1, 3, 2], [4, 5, 4], [6, 3, 3]], [[5, 2, 1], [
            5, 1, 1], [7, 2, 3]], [[4, 2, 5], [1, 9, 6], [4, 3, 1]]], dtype=torch.float32)
        self.vec = torch.tensor([1, 1, 1], dtype=torch.float32)
        self.vec_tens_prod = torch.tensor(93, dtype=torch.float32)
        self.assertTrue(torch.equal(tools.vec_tens_prod(
            self.vec, self.tens), self.vec_tens_prod))

    # TODO: Test for cut_rth_slice
    #
    # def test_pseudoLH_2DTensor(self):
    #     mod = MInteractionModel.MInteractionModel(order=2)
    #     arr = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
    #     df = pd.DataFrame(np.rot90(arr), columns=("X1", "X2"))
    #     mod.data = df
    #     mod.Q = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
    #     self.assertTrue(float(mod.pseudoLH(mod.Q)) == 4.184737682342529)
    #
    # def test_pseudoLH_3DTensor(self):
    #     mod = MInteractionModel.MInteractionModel(order=3)
    #     arr = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1,
    #                                                0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0]])
    #     df = pd.DataFrame(np.rot90(arr), columns=("X1", "X2", "X3"))
    #     mod.data = df
    #     mod.dim = len(df.columns)
    #     mod.Q = torch.ones([mod.dim] * mod.order, dtype=torch.float32)
    #     self.assertTrue(float(mod.pseudoLH(mod.Q)) == 12.98631477355957)


if __name__ == '__main__':
    unittest.main()
