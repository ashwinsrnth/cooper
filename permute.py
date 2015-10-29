import numpy as np
from numpy.testing import assert_allclose

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
    for i in range(n3):
        for j in range(n2):
            for k in range(n1):
                B.ravel()[i*n1*n2+j*n1+k] = A.ravel()[i*out_strides[0]+j*out_strides[1]+k*out_strides[2]]
    return B

A = np.random.rand(4, 5, 6)
assert_allclose(permute(A, (1, 2, 0)), A.transpose((1, 2, 0)).copy())

