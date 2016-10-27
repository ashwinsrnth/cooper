from pycuda import autoinit
from pycuda.autoinit import context
import pycuda.compiler as compiler
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

import numpy as np

with open('/home/atrikut/projects/cooper/cuTranspose/transpose3d.cu') as f:
     kernels = f.read()
mod = compiler.SourceModule(source=kernels, options=["-O3"], arch="sm_35", include_dirs=["/home/atrikut/projects/cooper/cuTranspose/"])

permute_210_inplace = mod.get_function("dev_transpose_210_in_place")
permute_210_inplace.prepare('Pii')

permute_102_inplace = mod.get_function("dev_transpose_102_in_place")
permute_102_inplace.prepare('Pii')

permute_021_inplace = mod.get_function("dev_transpose_021_in_place")
permute_021_inplace.prepare('Pii')

permute_201_inplace = mod.get_function("dev_transpose_201_in_place")
permute_201_inplace.prepare('Pi')

permute_120_inplace = mod.get_function("dev_transpose_120_in_place")
permute_120_inplace.prepare('Pi')

# Oneelement per thread:

permute_210_ept1 = mod.get_function("dev_transpose_210_ept1")
permute_210_ept1.prepare('PPiii')

permute_102_ept1 = mod.get_function("dev_transpose_102_ept1")
permute_102_ept1.prepare('PPiii')

permute_021_ept1 = mod.get_function("dev_transpose_021_ept1")
permute_021_ept1.prepare('PPiii')

permute_201_ept1 = mod.get_function("dev_transpose_201_ept1")
permute_201_ept1.prepare('PPiii')

permute_120_ept1 = mod.get_function("dev_transpose_120_ept1")
permute_120_ept1.prepare('PPiii')

# Four elements per thread:

permute_210_ept4 = mod.get_function("dev_transpose_210_ept4")
permute_210_ept4.prepare('PPiii')

permute_102_ept4 = mod.get_function("dev_transpose_102_ept4")
permute_102_ept4.prepare('PPiii')

permute_021_ept4 = mod.get_function("dev_transpose_021_ept4")
permute_021_ept4.prepare('PPiii')

permute_201_ept4 = mod.get_function("dev_transpose_201_ept4")
permute_201_ept4.prepare('PPiii')

permute_120_ept4 = mod.get_function("dev_transpose_120_ept4")
permute_120_ept4.prepare('PPiii')

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
def cuTranspose_permute_inplace(a_d, permutation):
    N = a_d.shape[0]
    if permutation == (2, 1, 0): #210
        permute_210_inplace.prepared_call((N/32, N/32, N), (32, 32, 1),
                a_d.gpudata, N, N)
    elif permutation == (0, 2, 1): #102
        permute_102_inplace.prepared_call((N/32, N/32, N), (32, 32, 1),
                a_d.gpudata, N, N)
    elif permutation == (1, 0, 2): #021
        permute_021_inplace.prepared_call((N/32, N/32, N), (32, 32, 1),
                a_d.gpudata, N, N)
    elif permutation == (1, 2, 0): #201
        permute_201_inplace.prepared_call((N/8, N/8, N/8), (8, 8, 8),
                a_d.gpudata, N)
    elif permutation == (2, 0, 1): #120
        permute_120_inplace.prepared_call((N/8, N/8, N/8), (8, 8, 8),
                a_d.gpudata, N)

def cuTranspose_permute_ept1(a_d, b_d, permutation):
    N = b_d.shape[0]
    if permutation == (2, 1, 0): #210
        permute_210_ept1.prepared_call((N/32, N/32, N), (32, 32, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (0, 2, 1): #102
        permute_102_ept1.prepared_call((N/32, N/32, N), (32, 32, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (1, 0, 2): #021
        permute_021_ept1.prepared_call((N/32, N/32, N), (32, 32, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (1, 2, 0): #201
        permute_201_ept1.prepared_call((N/32, N/32, N), (32, 32, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (2, 0, 1): #120
        permute_120_ept1.prepared_call((N/32, N/32, N), (32, 32, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (0, 1, 2):
        cuda.memcpy_dtod(b_d.gpudata, a_d.gpudata, b_d.nbytes)

def cuTranspose_permute_ept4(a_d, b_d, permutation):
    N = b_d.shape[0]
    if permutation == (2, 1, 0): #210
        permute_210_ept4.prepared_call((N/32, N/32, N), (32, 8, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (0, 2, 1): #102
        permute_102_ept4.prepared_call((N/32, N/32, N), (32, 8, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (1, 0, 2): #021
        permute_021_ept4.prepared_call((N/32, N/32, N), (32, 8, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (1, 2, 0): #201
        permute_201_ept4.prepared_call((N/32, N/32, N), (32, 8, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (2, 0, 1): #120
        permute_120_ept4.prepared_call((N/32, N/32, N), (32, 8, 1),
                b_d.gpudata, a_d.gpudata, N, N, N)
    elif permutation == (0, 1, 2):
        cuda.memcpy_dtod(b_d.gpudata, a_d.gpudata, b_d.nbytes)
