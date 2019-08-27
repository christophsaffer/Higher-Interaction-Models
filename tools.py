import torch
import itertools
import time


def vec_tens_prod(vec, tens):

    for i in range(0, tens.dim()):
        tens = torch.matmul(vec, tens)

    return tens


def make_tens_symm(tens):
    symm_tens = torch.zeros([len(tens)] * tens.dim())
    permutations = list(itertools.permutations(range(0, tens.dim())))
    for x in permutations:
        symm_tens += tens.permute(x)

    symm_tens = symm_tens/len(permutations)

    return symm_tens


def make_tens_super_symm(tens):
    product = list(itertools.product(range(len(tens)), repeat=tens.dim()))
    completo = []
    for x in product:
        permutations = list(itertools.permutations(x))
        completo.append(set(permutations))
    output = set()
    for x in completo:
        output.add(frozenset(x))

    result = []
    combs = []
    for y in output:
        temp = []
        z = list(y)
        for x in y:
            temp.append(set(x))

        combs.append(tuple(temp[0]))
        result.append(z)
    combs_new = list(set(combs))
    final_list = []
    for x in combs_new:
        li = []
        for i in range(len(result)):
            if x == combs[i]:
                li.append(result[i])

        final_list.append(sum(li, []))

    symm_tens = torch.zeros([len(tens)] * tens.dim())
    for multi in final_list:
        s = 0
        for x in multi:
            s += tens[x]
        for x in multi:
            symm_tens[x] = s/len(multi)

    return symm_tens


def cut_rth_slice(tens, r):

    slices = []
    sli = tens
    for k in range(0, tens.dim()):
        sli = torch.unbind(sli, dim=0)[r]
        slices.insert(0, sli)

    return slices


def nuclear_norm_tens(tens):

    if tens.dim() > 2:
        flattend = tens.reshape((len(tens), len(tens)**2))
        return torch.nuclear_norm(flattend)
    else:
        return torch.nuclear_norm(tens)


def measure_time(iter, func, *args):
    summe = 0
    start = time.time()
    for n in range(iter):
        func(*args)
    ende = time.time()
    print('{:5.3f}s'.format(ende-start))


# def set_values_to_tensor(q):
#     if len(q) == 7:
#         q1, q2, q3, q12, q13, q23, q123 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
#         tens = torch.tensor([[[q1, q12, q13], [q12, q12, q123], [q13, q123, q13]], [[q12, q12, q123], [q12, q2, q23], [
#                             q123, q23, q23]], [[q13, q123, q13], [q123, q23, q23], [q13, q23, q3]]], dtype=torch.float32, requires_grad=True)
#         return tens
#     return q
