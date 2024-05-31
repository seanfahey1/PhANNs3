import numpy as np
from scipy.stats import zscore


def zscore(array):
    """
    Calculates the mean, standard deviation, and zscore arrays for PhANNs data loading.
    """
    print(array.shape)
    mean_array = np.mean(array, axis=1)
    stddev_array = np.std(array, axis=1)
    zscore_array = zscore(array, axis=1)
    print(mean_array.shape)
    print(zscore_array.shape)
    return mean_array, stddev_array, zscore_array
