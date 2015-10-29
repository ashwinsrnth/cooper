import numpy as np
from numpy.testing import assert_allclose
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import pycuda.compiler as compiler
from pycuda.tools import context_dependent_memoize

kernel_text = '''
__global__ void permuteKernel(double *in_d,
            double *out_d,
            const int n3,
            const int n2,
            const int n1,
            const int x_stride,
            const int y_stride,
            const int z_stride) {
    int tix = blockDim.x*blockIdx.x + threadIdx.x;
    int tiy = blockDim.x*blockIdx.y + threadIdx.y;
    int tiz = blockDim.x*blockIdx.z + threadIdx.z;
    out_d[tiz*n1*n2 + tiy*n1 + tix] = \
        in_d[tiz*z_stride + tiy*y_stride + tix*x_stride];
}
'''

@context_dependent_memoize
def _get_permute_kernel():
    module = compiler.SourceModule(kernel_text,
            options=['-O2'])
    permute_kernel = module.get_function(
            'permuteKernel')
    permute_kernel.prepare('PPiiiiii')
    return permute_kernel

def permute(a, b, permutation):
    '''
    :param a:
    :type a:
    :param b:
    :type b:
    :param permutation: 
    '''
    a_strides = np.array(A.strides)/A.dtype.itemsize
    strides = a_strides[list(permutation)]
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    f = _get_permute_kernel()
    f.prepared_call((b.shape[2]/8, b.shape[1]/8, b.shape[0]/8),
            (8, 8, 8),
            a_d.gpudata, b_d.gpudata,
            b.shape[2], b.shape[1], b.shape[0],
            strides[2], strides[1], strides[0])
    b[...] = b_d.get()

A = np.random.rand(32, 32, 32)
B = np.zeros((32, 32, 32), dtype=np.float64)
permute(A, B, (1, 2, 0))
assert_allclose(B, A.transpose((1, 2, 0)).copy())

