import numpy as np

cimport numpy as np


def zscore(np.ndarray[np.float64_t, ndim=2] array):
    """
    Calculates the mean, standard deviation, and zscore arrays for PhANNs data loading.

    note: to compile, run `python setup.py build_ext --inplace`
    """
    cdef np.ndarray[np.float64_t, ndim=1] mean_array = np.zeros(array.shape[1])
    cdef np.ndarray[np.float64_t, ndim=1] stddev_array = np.zeros(array.shape[1])
    cdef np.ndarray[np.float64_t, ndim=2] zscore_array = np.zeros((array.shape[0], array.shape[1]))

    cdef int i, j
    cdef double mean_val, stddev_val, zscore_val

    # loop through columns
    for i in range(array.shape[1]):
        mean_val = np.mean(array[:, i])  # column mean
        stddev_val = np.std(array[:, i])  # column std dev

        mean_array[i] = mean_val
        stddev_array[i] = stddev_val

        # loop through rows
        for j in range(array.shape[0]):
            if stddev_val != 0:
                zscore_val = (array[j, i] - mean_val) / stddev_val
            else:
                zscore_val = 0

            zscore_array[j, i] = zscore_val

    return mean_array, stddev_array, zscore_array
