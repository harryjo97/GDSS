import concurrent.futures
from functools import partial

import numpy as np
import pyemd
from scipy.linalg import toeplitz

from sklearn.metrics.pairwise import pairwise_kernels
from evaluation.eden import vectorize

# NOTES:
# EMD stands for earth move distance, i.e. Wasserstein metric,
# (\inf_{\gama \in \Gama(\mu, \nu) \int_{M*M} d(x,y)^p d\gama(x,y))^(1/p)
# -------- From Niu et al. (2020) --------

def emd(x, y, distance_scaling=1.0):
    # -------- convert histogram values x and y to float, and make them equal len --------
    x = x.astype(np.float)
    y = y.astype(np.float)
    support_size = max(len(x), len(y))
    # -------- diagonal-constant matrix --------
    d_mat = toeplitz(range(support_size)).astype(np.float)  
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """ Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    x = x.astype(np.float)
    y = y.astype(np.float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    # -------- convert histogram values x and y to float, and make them equal len --------
    x = x.astype(np.float)
    y = y.astype(np.float)
    x, y = process_tensor(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """ Discrepancy between 2 samples
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker,
                                     [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                d += dist
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """ MMD between two samples
    """
    # -------- normalize histograms into pmf --------
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
        disc(samples2, samples2, kernel, *args, **kwargs) - \
        2 * disc(samples1, samples2, kernel, *args, **kwargs)


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """ EMD between average of two samples
    """

    # -------- normalize histograms into pmf --------
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def process_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y
