import numpy as np
import torch
import itertools


def tens_vec_prod(vec, tens):

    for i in range(0, tens.dim()):
        tens = torch.matmul(vec, tens)

    return tens


def make_tens_symm(tens):
    # tens = torch.tensor(tens, dtype=torch.float64)
    symm_tens = torch.zeros([len(tens)] * tens.dim())
    permutations = list(itertools.permutations(range(0, tens.dim())))
    for x in permutations:
        symm_tens += tens.permute(x)

    return symm_tens/len(permutations)
