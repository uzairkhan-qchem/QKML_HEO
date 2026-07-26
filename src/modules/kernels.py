import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
import gpflow


def angular_rbf_kernel_function(x, y):
    return np.exp(-(np.arccos(1 - cosine_distance(x, y)) ** 2))


def cosine_kernel_function(x, y):
    return 1 - cosine_distance(x, y)


def cosine_rbf_kernel_function(x, y):
    return np.exp(-(cosine_distance(x, y) ** 2))


def cosine_distance_exponential_kernel_function(x, y):
    return np.exp(-cosine_distance(x, y))


def gaussian_rbf_euclidean_kernel_function(x, y, lengthscale):
    return np.exp(-((np.linalg.norm(x - y) / lengthscale) ** 2))


class FixedPrecomputedGPKernel(gpflow.kernels.Kernel):
    """
    A GPflow-compatible kernel that uses a precomputed, fixed kernel matrix.
    """

    def __init__(self, kernel_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_matrix = kernel_matrix

    def K(self, X, X2=None):
        if not isinstance(X, np.ndarray):
            X = X.numpy()  # type: ignore

        X = X.flatten().astype(int)

        if X2 is None:
            indices = np.ix_(X, X)
        else:
            if not isinstance(X2, np.ndarray):
                X2 = X2.numpy()  # type: ignore

            X2 = X2.flatten().astype(int)

            indices = np.ix_(X, X2)
        return self.kernel_matrix[indices]

    def K_diag(self, X):  # type: ignore
        if not isinstance(X, np.ndarray):
            X = X.numpy()
        X = X.flatten().astype(int)
        return np.ones(X.shape[0])  # type: ignore