from pycuda import autoinit
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from cuper import permute
import itertools

def time_permute(a_d, permutation, b_d=None, impl=None):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for i in range(100):
        permute(a_d, permutation, b_d, impl)
    end.record()
    end.synchronize()
    time_taken = start.time_till(end)/100000.
    return time_taken

print
print
print("Time (in ms) for performing permutations - averaged over 100 runs")
print("-----------------------------------------------------------------")
print
for N in [32, 64, 128, 256, 512]:
    for permutation in itertools.permutations((0, 1, 2)):
        a_d = gpuarray.to_gpu(np.random.rand(N, N, N))
        b_d = gpuarray.to_gpu(np.random.rand(N, N, N))
        time_naive = time_permute(a_d, permutation, b_d, impl="naive")
        time_cut = time_permute(a_d, permutation, b_d, impl="cuTranspose")
        if permutation == (0, 1, 2):
            pass
        else:
            print("Size: {:4} | Permutation: {} | Naive: {: .6f} | cuTranspose: {: .6f} | Speedup: {: .6f}".format(
                N, permutation, time_naive, time_cut, time_naive/time_cut))
    print
