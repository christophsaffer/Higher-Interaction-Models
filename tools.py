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
        sli = torch.unbind(sli, dim=0)[r]
        slices.insert(0, sli)

    return slices


def nuclear_norm_tens(tens):

    flattend = tens.reshape((len(tens), len(tens)**2))
    return torch.nuclear_norm(flattend)
