import numpy as np
import torch
import itertools


def vec_tens_prod(vec, tens):

    vec = torch.tensor(vec, dtype=torch.float32)

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

    slices = []
    sli = tens
    for k in range(0, tens.dim()):
        sli = torch.unbind(tens, dim=0)[r]
        slices.append(sli)

    return slices


def nuclear_norm_tens(tens):

    flattend = tens.reshape((len(tens), len(tens)**2))
    return torch.nuclear_norm(flattend)


# def cut_rth_slice_ord3(tens, r):
#     slices2d = []
#     slices1d = []
#     slices0d = []
#     for i in range(0, tens.dim()):
#         slice = torch.unbind(tens, dim=i)[r]
#         slices2d.append(slice)
#         if ((-1)**i == 1):
#             slices1d.append(slice[r])
#         else:
#             slices1d.append(slice.t()[r])
#
#     slices0d.append(slice[r][r])
#
#     return slices2d, slices1d, slices0d
#
#
# def cut_rth_slice(tens, r):
#     slices = []
#     for j in range(0, tens.dim()):
#         for i in range(0, tens.dim()):
#             slice = torch.unbind(tens, dim=i)[r]
#             for k in range(0, j):
#                 if ((-1)**i == 1):
#                     slice = slice[r]
#                 else:
#                     slice = slice.t()[r]
#             if not tensorinlist(slices, slice):
#                 slices.append(slice)
#     return slices


def tensorinlist(list, tens):
    for x in list:
        if torch.equal(x, tens):
            return True

    return False
