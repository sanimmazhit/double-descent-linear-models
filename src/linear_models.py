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


def ridge_regression(X, y, alpha=1.0):
    """
    Compute Ridge Regression coefficients using
    L2 regularization.

    Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        alpha: Regularization strength

    Returns:
        Model coefficients
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    n_features = X.shape[1]

    # Create identity matrix
    identity = np.eye(n_features)

    # Calculate ridge regression coefficients
    weights = np.linalg.solve(
        X.T @ X + alpha * identity,
        X.T @ y
    )

    return weights
