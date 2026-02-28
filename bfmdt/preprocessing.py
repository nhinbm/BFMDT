import numpy as np


class Preprocessor:
    """Clean and normalize raw data for the pipeline.

    Args:
        X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Raw decision labels of shape (n_samples,).

    Returns:
        X_clean (np.ndarray): Normalized feature matrix in [0, 1], missing values imputed. Shape (n_samples, n_features).
        y_clean (np.ndarray): Ordinal integer labels, samples with missing decisions removed. Shape (n_samples,).
    """

    def __init__(self):
        self.feature_min_ = None
        self.feature_max_ = None

    def fit_transform(self, X, y):
        # TODO: implement
        raise NotImplementedError

    def transform(self, X):
        # TODO: implement
        raise NotImplementedError
