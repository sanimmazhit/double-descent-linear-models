import numpy as np


def generate_synthetic_data(n_samples=100, n_features=20,
                            noise_std=0.5, random_state=42):
    """
    Generate a synthetic dataset for linear regression.
    """
    rng = np.random.default_rng(random_state)

    # Generate random input features
    X = rng.normal(0, 1, size=(n_samples, n_features))

    # Generate true model coefficients
    true_weights = rng.normal(0, 1, size=n_features)

    # Add Gaussian noise to the target values
    noise = rng.normal(0, noise_std, size=n_samples)

    # Generate target values using a linear model
    y = X @ true_weights + noise

    return X, y


if __name__ == "__main__":
    X, y = generate_synthetic_data()

    print("Synthetic dataset generated successfully!")
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("First five target values:", y[:5])