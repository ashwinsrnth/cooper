from pycuda import autoinit
from pycuda.autoinit import context
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

import numpy as np

with open('/home/atrikut/projects/cooper/cuTranspose/transpose3d.cu') as f:
     kernels = f.read()
mod = compiler.SourceModule(source=kernels, options=["-O2"], arch="sm_35", include_dirs=["/home/atrikut/projects/cooper/cuTranspose/"])

permute_210 = mod.get_function("dev_transpose_210_in_place")
permute_210.prepare('Pii')

permute_102 = mod.get_function("dev_transpose_102_in_place")
permute_102.prepare('Pii')

permute_021 = mod.get_function("dev_transpose_021_in_place")
permute_021.prepare('Pii')

permute_201 = mod.get_function("dev_transpose_201_in_place")
permute_201.prepare('Pi')

permute_120 = mod.get_function("dev_transpose_120_in_place")
permute_120.prepare('Pi')

"""
The discrepancy in the indices is that
numpy thinks of the "0" axis as being the z (slowest) axis,
while cuTranspose think of the "0" axis as being the x (fastest) axis.
To switch between the triplet notations,
change 2->0; 0->2 and flip the triplet (left to right).
e.g., 210 -> 210
      102 -> 021
      etc.,
"""
def cuTranspose_permute(a_d, b_d, permutation):
    N = b_d.shape[0]
    if permutation == (2, 1, 0): #210
        permute_210.prepared_call((N/16, N/16, N), (16, 16, 1),
                a_d.gpudata, N, N)
    elif permutation == (0, 2, 1): #102
        permute_102.prepared_call((N/16, N/16, N), (16, 16, 1),
                a_d.gpudata, N, N)
    elif permutation == (1, 0, 2): #021
        permute_021.prepared_call((N/16, N/16, N), (16, 16, 1),
                a_d.gpudata, N, N)
    elif permutation == (1, 2, 0): #201
        permute_201.prepared_call((N/8, N/8, N/8), (8, 8, 8),
                a_d.gpudata, N)
    elif permutation == (2, 0, 1): #120
        permute_120.prepared_call((N/8, N/8, N/8), (8, 8, 8),
                a_d.gpudata, N)
    cuda.memcpy_dtod(b_d.gpudata, a_d.gpudata, a_d.nbytes)
