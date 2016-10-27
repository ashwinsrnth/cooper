from naive.permute import naive_permute
from cuTranspose.permute import cuTranspose_permute_ept1, cuTranspose_permute_inplace

def permute(a_d, permutation, b_d=None, impl="naive"):
    '''
    Permute the data in a 3-dimensional
    GPUArray

    :param a_d: Array containing data to permute
    :type a_d: pycuda.gpuarray.GPUArray
    :param permutation: The desired permutation of the axes
        of a_d
    :type permutation: list or tuple
    :param b_d: Space for output (if not inplace)
    :type b_d: pycuda.gpuarray.GPUArray
    :param impl: Name of implementation to use (naive, cuTranspose)
    :type impl: str

    If b_d is unspecified, then the transpose is done inplace
    '''
    if b_d is None:
        cuTranspose_permute_inplace(a_d, permutation)
    else:
        if impl == "naive":
            naive_permute(a_d, b_d, permutation)
        if impl == "cuTranspose":
            cuTranspose_permute_ept1(a_d, b_d, permutation)
