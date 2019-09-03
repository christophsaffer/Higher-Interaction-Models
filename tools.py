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


def get_str_symm_idx_lst(tens):
    # Get the list of lists of strongly symmetric indexes of the input tensor

    # Get all indexes of input tensor, i.e. order=3, dim=4 [(000), (100), ..., (333)]
    prod = list(itertools.product(range(len(tens)), repeat=tens.dim()))

    # Collect indexes with same numbers, i.e. order=3, dim=4 {{(1,2,2), (2,1,2), (2,2,1)},{(1,1,1)}, ...}
    # Using 'set', 'frozenset' instead of list because duplicate entries are not possible
    prod_grouped_set = set()
    for x in prod:
        permutations = list(itertools.permutations(x))
        prod_grouped_set.add(frozenset(permutations))

    # Convert the set of frozensets (prod_grouped_set) to a list of lists (prod_grouped)
    prod_grouped = []
    combs = []
    for y in prod_grouped_set:
        combs.append(tuple(set(list(y)[0])))
        prod_grouped.append(list(y))

    # Get possible combinations of unique indexes
    combs_unique = list(set(combs))

    # Create grouped indexes list from prod_group and combs_unique
    grouped_idx = []
    for x in combs_unique:
        li = []
        for i in range(len(prod_grouped)):
            if x == combs[i]:
                li.append(prod_grouped[i])

        grouped_idx.append(sum(li, []))

    return grouped_idx


def make_tens_str_symm(tens, grouped_idx=None):
    # Make input tens strongly symmetric, grouped_idx from get_str_symm_idx_lst function necessary

    if grouped_idx is None:
        grouped_idx = get_str_symm_idx_lst(tens)

    for symm_idx in grouped_idx:
        mean = 0
        for x in symm_idx:
            mean += tens[x]
        mean /= len(symm_idx)
        for x in symm_idx:
            tens[x] = mean
    return tens


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


def save_3Dtens_to_file(tens, filename="tensor.txt", digits=8):

    f = open(filename, "w+")
    f.write(str(len(tens)) + ", ")
    for i in range(0, len(tens)):
        for j in range(0, len(tens)):
            for k in range(0, len(tens)):
                f.write(str(round(float(tens[k, j, i]), digits)) + ", ")

    f.close()
# def set_values_to_tensor(q):
#     if len(q) == 7:
#         q1, q2, q3, q12, q13, q23, q123 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
#         tens = torch.tensor([[[q1, q12, q13], [q12, q12, q123], [q13, q123, q13]], [[q12, q12, q123], [q12, q2, q23], [
#                             q123, q23, q23]], [[q13, q123, q13], [q123, q23, q23], [q13, q23, q3]]], dtype=torch.float32, requires_grad=True)
#         return tens
#     return q
