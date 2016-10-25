from naive.permute import naive_permute

def permute(a_d, b_d, permutation, impl="naive"):
    '''
    Permute the data in a 3-dimensional
    GPUArray

    :param a_d: Array containing data to permute
    :type a_d: pycuda.gpuarray.GPUArray
    :param b_d: Space for output
    :type b_d: pycuda.gpuarray.GPUArray
    :param permutation: The desired permutation of the axes
        of a_d
    :type permutation: list or tuple
    '''
    
    if impl == "naive":
        naive_permute(a_d, b_d, permutation)
