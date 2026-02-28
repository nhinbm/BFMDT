import numpy as np


def count_inversions(arr):
    """Count inversions in array using merge sort. O(n log n).

    Args:
        arr (np.ndarray): 1D array of comparable values.

    Returns:
        n_inversions (int): Number of pairs (i, j) where i < j but arr[i] > arr[j].
    """
    # TODO: implement
    raise NotImplementedError


def compute_inversion_count_for_feature(X, y, feature_index):
    """Sort samples by feature value, count inversions in decision values.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision values of shape (n_samples,).
        feature_index (int): Which feature column to sort by.

    Returns:
        n_inversions (int): Inversion count of decision values when sorted by feature.
    """
    # TODO: implement
    raise NotImplementedError


def map_classes_to_ordinal(y):
    """Map arbitrary class labels to contiguous integers 0, 1, ..., K-1.

    Args:
        y (np.ndarray): Raw class labels of shape (n_samples,).

    Returns:
        y_mapped (np.ndarray): Integer-encoded labels of shape (n_samples,).
        label_map (dict): Mapping from original label to encoded integer.
    """
    # TODO: implement
    raise NotImplementedError
