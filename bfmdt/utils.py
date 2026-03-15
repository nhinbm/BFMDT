import numpy as np


def count_inversions(arr):
    """Count inversions in array using merge sort. O(n log n).

    Args:
        arr (np.ndarray): 1D array of comparable values.

    Returns:
        n_inversions (int): Number of pairs (i, j) where i < j but arr[i] > arr[j].
    """
    if len(arr) <= 1:
        return 0

    mid = len(arr) // 2
    left = arr[:mid].copy()
    right = arr[mid:].copy()

    inversions = count_inversions(left) + count_inversions(right)

    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            inversions += len(left) - i
            j += 1
        k += 1

    arr[k:k + len(left) - i] = left[i:]
    arr[k + len(left) - i:] = right[j:]

    return inversions


def compute_inversion_count_for_feature(X, y, feature_index):
    """Sort samples by feature value, count inversions in decision values.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision values of shape (n_samples,).
        feature_index (int): Which feature column to sort by.

    Returns:
        n_inversions (int): Inversion count of decision values when sorted by feature.
    """
    sorted_indices = np.lexsort((y, X[:, feature_index]))
    y_sorted = y[sorted_indices].astype(float).copy()
    return count_inversions(y_sorted)


def count_tied_pairs(X, feature_index):
    """Count pairs of samples with identical feature values.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        feature_index (int): Which feature column to check.

    Returns:
        n_tied (int): Number of pairs (i, j) where i < j and
            X[i, feature_index] == X[j, feature_index].
    """
    _, counts = np.unique(X[:, feature_index], return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def map_classes_to_ordinal(y):
    """Map arbitrary class labels to contiguous integers 0, 1, ..., K-1.

    Args:
        y (np.ndarray): Raw class labels of shape (n_samples,).

    Returns:
        y_mapped (np.ndarray): Integer-encoded labels of shape (n_samples,).
        label_map (dict): Mapping from original label to encoded integer.
    """
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y], dtype=int)
    return y_mapped, label_map
