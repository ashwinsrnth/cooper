import numpy as np
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
from numpy.testing import assert_allclose
from permute import permute

a = np.random.rand(32, 32, 32)
b = np.zeros((32, 32, 32), dtype=np.float64)
a_d = gpuarray.to_gpu(a)
b_d = gpuarray.to_gpu(b)
permute(a_d, b_d, (1, 2, 0))
assert_allclose(b_d.get(), a.transpose((1, 2, 0)).copy())
