import numpy as np


class MonotonicPartitioner:
    """Partition samples into Maximal Monotonic Intervals (MMIs) per feature.

    Args:
        X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Normalized decision values of shape (n_samples,).

    Returns:
        ascending_partitions (list[list[np.ndarray]]): AMP per feature. Each feature has a list of MMIs, each MMI is an array of sample indices.
        descending_partitions (list[list[np.ndarray]]): DMP per feature. Same structure.
    """

    def __init__(self):
        self.ascending_partitions = None
        self.descending_partitions = None

    def fit(self, X, y):
        # TODO: implement Algorithm 1
        raise NotImplementedError
