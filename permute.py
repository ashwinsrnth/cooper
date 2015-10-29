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

def permute(A, permutation):
    '''
    Permute the values of 3D array A
    according to the permutation
    of axes specified
    '''
    in_strides = np.array(A.strides)/A.dtype.itemsize

    out_shape = (A.shape[permutation[0]], 
                A.shape[permutation[1]],
                A.shape[permutation[2]])
    out_strides = (in_strides[permutation[0]],
                    in_strides[permutation[1]],
                    in_strides[permutation[2]])
    B =  np.zeros(out_shape, A.dtype)

    n3, n2, n1 = out_shape
    A_d = gpuarray.to_gpu(A)
    B_d = gpuarray.to_gpu(B)
    f = _get_permute_kernel()
    f.prepared_call((n1/4, n2/4, n3/4), (4, 4, 4),
            A_d.gpudata, B_d.gpudata, n1, n2, n3, out_strides[2],
            out_strides[1], out_strides[0])
    B = B_d.get()
    return B

A = np.random.rand(32, 32, 32)
assert_allclose(permute(A, (1, 2, 0)), A.transpose((1, 2, 0)).copy())

