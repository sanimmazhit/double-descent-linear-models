
import numpy as np


def generate_synthetic_data(n_samples=100, n_features=20,
                            noise_std=0.5, random_state=42):
    """Generate a synthetic linear regression dataset."""

    rng = np.random.default_rng(random_state)

    # Generate features and true coefficients
    X = rng.normal(0, 1, size=(n_samples, n_features))
    true_weights = rng.normal(0, 1, size=n_features)

    # Generate target values with Gaussian noise
    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ true_weights + noise

    return X, y


def generate_double_descent_data(
    n_train=80,
    n_test=500,
    max_features=200,
    noise_std=0.5,
    random_state=42
):
    """Generate fixed data with signal in the first 20 features."""

    rng = np.random.default_rng(random_state)

    # Generate training and test features
    X_train = rng.normal(size=(n_train, max_features))
    X_test = rng.normal(size=(n_test, max_features))

    # True signal uses the first 20 features
    true_weights = np.zeros(max_features)
    true_weights[:20] = rng.normal(size=20)

    # Generate noisy linear targets
    y_train = X_train @ true_weights + rng.normal(
        scale=noise_std, size=n_train
    )
    y_test = X_test @ true_weights + rng.normal(
        scale=noise_std, size=n_test
    )

    return X_train, X_test, y_train, y_test


def generate_dense_signal_data(
    n_train=80,
    n_test=500,
    max_features=500,
    noise_std=0.5,
    random_state=42
):
    """Generate regression data with signal across all features."""

    rng = np.random.default_rng(random_state)

    # Generate training and test features
    X_train = rng.normal(size=(n_train, max_features))
    X_test = rng.normal(size=(n_test, max_features))

    # Distribute signal across all features
    true_weights = rng.normal(size=max_features)
    true_weights /= np.linalg.norm(true_weights)

    # Generate noisy linear targets
    y_train = X_train @ true_weights + rng.normal(
        scale=noise_std, size=n_train
    )
    y_test = X_test @ true_weights + rng.normal(
        scale=noise_std, size=n_test
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = generate_synthetic_data()

    print("Synthetic dataset generated successfully!")
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("First five target values:", y[:5])
