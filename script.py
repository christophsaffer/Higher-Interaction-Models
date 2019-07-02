import numpy as np

import Ising_MOD

testmod = Ising_MOD.Ising_MOD()
testmod.add_dataset("../cgmodelselection/unittest_data/py_D_s12d3l0.csv")
testmod.modelselection()
