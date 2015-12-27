# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from pycuda import autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from permute import permute
import itertools

def time_permute():
    a = np.random.rand(32, 32, 32)
    b = np.zeros((32, 32, 32), dtype=np.float64)
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    permute(a_d, b_d, permutation)
    assert_allclose(b_d.get(), a.transpose(permutation).copy())

