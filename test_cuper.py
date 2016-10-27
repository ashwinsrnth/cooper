from __future__ import print_function
import numpy as np
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
from numpy.testing import assert_allclose
from cuper import permute 
import itertools

for impl in ['naive', 'cuTranspose']:
    for permutation in itertools.permutations((0, 1, 2)):
        a = np.random.rand(32, 32, 32)
        b = np.zeros((32, 32, 32), dtype=np.float64)
        a_d = gpuarray.to_gpu(a)
        b_d = gpuarray.to_gpu(b)
        print("Testing {} implementation for permutation {} ...".format(impl, permutation), end="")
        permute(a_d, permutation, b_d, impl=impl)
        try:
            assert_allclose(b_d.get(), a.transpose(permutation).copy())
            print("Success.")
        except AssertionError:
            print(".................Failure.")

for permutation in itertools.permutations((0, 1, 2)):
    a = np.random.rand(32, 32, 32)
    b = np.zeros((32, 32, 32), dtype=np.float64)
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    print("Testing cuTranspose (inplace) implementation for permutation {} ...".format(permutation), end="")
    permute(a_d, permutation, impl="cuTranspose")
    try:
        assert_allclose(a_d.get(), a.transpose(permutation).copy())
        print("Success.")
    except AssertionError:
        print("...............Failure.")

