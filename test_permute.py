from __future__ import print_function
import numpy as np
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
from numpy.testing import assert_allclose
from permute import permute
import itertools

for permutation in itertools.permutations((0, 1, 2)):
    a = np.random.rand(32, 32, 32)
    b = np.zeros((32, 32, 32), dtype=np.float64)
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    permute(a_d, b_d, permutation, impl="naive")
    print("Testing permutation {} ...".format(permutation), end="")
    assert_allclose(b_d.get(), a.transpose(permutation).copy())
    print("Success.")
