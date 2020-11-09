import numpy as np

def np_unique_axis0(array):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[np.lexsort(array.T[::-1])]
    mask        = np.empty(array.shape[0], dtype=np.bool_)
    mask[0]     = True
    mask[1:]    = np.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

x = np.array(range(25), dtype=np.int32).reshape((5,5))
x[1] = x[0]
x[4] = x[2]
print(x)
print(np_unique_axis0(x))