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
        """Compute ascending and descending monotonic partitions for all features.

        Args:
            X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Normalized decision values of shape (n_samples,).

        Returns:
            self: Fitted partitioner with ascending_partitions and
                descending_partitions populated.
        """
        _, n_features = X.shape

        self.ascending_partitions = []
        self.descending_partitions = []

        for j in range(n_features):
            self.ascending_partitions.append(
                self._partition_feature(X, y, j, direction="ascending")
            )
            self.descending_partitions.append(
                self._partition_feature(X, y, j, direction="descending")
            )

        return self

    @staticmethod
    def _partition_feature(X, y, feature_index, direction):
        """Partition one feature into Maximal Monotonic Intervals.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Decision values of shape (n_samples,).
            feature_index (int): Which feature column to partition by.
            direction (str): 'ascending' for non-decreasing MMIs,
                'descending' for non-increasing MMIs.

        Returns:
            mmis (list[np.ndarray]): List of MMIs. Each MMI is an array
                of original sample indices.
        """
        n_samples = X.shape[0]

        if n_samples <= 1:
            return [np.arange(n_samples)]

        sorted_indices = np.lexsort((y, X[:, feature_index]))

        y_sorted = y[sorted_indices]
        if direction == "ascending":
            breaks = y_sorted[:-1] > y_sorted[1:]
        else:
            breaks = y_sorted[:-1] < y_sorted[1:]

        split_points = np.where(breaks)[0] + 1
        mmis = np.split(sorted_indices, split_points)

        return list(mmis)
