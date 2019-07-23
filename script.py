import numpy as np
import pandas as pd
import itertools

import Ising_MOD

testmod = Ising_MOD.Ising_MOD()
# testmod.add_dataset("../cgmodelselection/unittest_data/py_D_s12d3l0.csv")
testmod.add_dataset("../syn_dat01.csv")
testmod.modelselect()

print(testmod.parameters.reshape((2, 2)))
# print(testmod.data.pivot_table(index=['X0', 'X1', 'X2'], aggfunc='size'))
print(testmod.data.pivot_table(index=['X', 'Y'], aggfunc='size'))

li = list(itertools.product([0, 1], repeat=2))

for x in li:
    print(x, ": ", testmod.funcvalue(x))

# print(testmod.parameters.reshape((3, 3)))
# print(testmod.parameters.reshape((2, 2)))

print(testmod.pseudoLH(testmod.parameters))
