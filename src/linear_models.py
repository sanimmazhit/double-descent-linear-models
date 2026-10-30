import numpy as np


def least_squares(X, y):
    """
    Compute the minimum-norm least squares solution.

    Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)

    Returns:
        Model coefficients
    """
    weights = np.linalg.pinv(X) @ y

    return weights