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


def cut_rth_slice(tens, r):
    slices2d = []
    slices1d = []
    slices0d = []
    for i in range(0, tens.dim()):
        slice = torch.unbind(d, dim=i)[r]
        slices2d.append(slice)
        if ((-1)**i == 1):
            slices1d.append(slice[1])
        else:
            slices1d.append(slice.t()[1])

    slices0d.append(slice[r][r])

    return slices2d, slices1d, slices0d
