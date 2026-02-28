import numpy as np


class FittingDegreeComputer:
    """Measure how well each sample fits the monotonic relationship under each feature.

    Args:
        X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Normalized decision values of shape (n_samples,).
        ascending_partitions (list[list[np.ndarray]]): AMP from MonotonicPartitioner.
        descending_partitions (list[list[np.ndarray]]): DMP from MonotonicPartitioner.

    Returns:
        matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features). Values in [0, 1].
        monotone_directions (np.ndarray): Direction per feature of shape (n_features,). +1 (increasing) or -1 (decreasing).
        X_adjusted (np.ndarray): Feature matrix with decreasing features inverted. Shape (n_samples, n_features).
    """

    def __init__(self):
        self.matrix = None
        self.monotone_directions = None
        self.X_adjusted = None

    def fit(self, X, y, ascending_partitions, descending_partitions):
        # TODO: implement Algorithm 2
        raise NotImplementedError
