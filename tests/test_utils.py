import numpy as np

from bfmdt.utils import (
    count_inversions,
    compute_inversion_count_for_feature,
    map_classes_to_ordinal,
)


# --- count_inversions ---


def test_count_inversions_zero():
    arr = np.array([1, 2, 3, 4, 5])
    assert count_inversions(arr) == 0


def test_count_inversions_nonzero():
    arr = np.array([5, 4, 3, 2, 1])
    assert count_inversions(arr) == 10


# --- compute_inversion_count_for_feature ---


def test_compute_inversion_tied_features_different_labels():
    """Tied feature values should be sorted by decision (Alg 1, Step 4)."""
    X = np.array([[1.0], [1.0], [2.0]])
    y = np.array([3, 1, 2])
    # Sorted by feature asc: y becomes [1, 3, 2] → 1 inversions
    assert compute_inversion_count_for_feature(X, y, 0) == 1


def test_compute_inversion_same_y_as_count():
    """Same reversed y, but accessed through feature sorting."""
    X = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])
    y = np.array([5, 4, 3, 2, 1])
    # Sorted by feature asc: y becomes [1, 2, 3, 4, 5] → 0 inversions
    assert compute_inversion_count_for_feature(X, y, 0) == 0




# --- map_classes_to_ordinal ---


def test_map_classes_to_ordinal():
    y = np.array(['high', 'low', 'medium', 'low', 'high'])
    y_mapped, label_map = map_classes_to_ordinal(y)
    assert label_map == {'high': 0, 'low': 1, 'medium': 2}
    np.testing.assert_array_equal(y_mapped, [0, 1, 2, 1, 0])