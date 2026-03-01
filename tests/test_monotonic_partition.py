import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner


@pytest.fixture
def partitioned():
    """X=[1,2,3,4], y=[1,3,2,4] -- has breaks in both directions."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1, 3, 2, 4])
    return MonotonicPartitioner().fit(X, y)


# --- ascending: y_sorted=[1,3,2,4], break at 3>2 -> MMIs: [0,1], [2,3] ---


def test_ascending_mmi_count(partitioned):
    assert len(partitioned.ascending_partitions[0]) == 2


def test_ascending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][0], [0, 1])


def test_ascending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][1], [2, 3])


# --- descending: y_sorted=[1,3,2,4], breaks at 1<3 and 2<4 -> MMIs: [0], [1,2], [3] ---


def test_descending_mmi_count(partitioned):
    assert len(partitioned.descending_partitions[0]) == 3


def test_descending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][0], [0])


def test_descending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][1], [1, 2])


def test_descending_third_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][2], [3])


# --- edge cases ---


def test_single_sample():
    mp = MonotonicPartitioner().fit(np.array([[5.0]]), np.array([1]))
    assert len(mp.ascending_partitions[0]) == 1


def test_tied_features_one_mmi_both_directions():
    """Tied features: ties broken by y -> one MMI in both directions."""
    mp = MonotonicPartitioner().fit(np.array([[1.0], [1.0], [1.0]]), np.array([3, 1, 2]))
    assert len(mp.ascending_partitions[0]) == 1
    assert len(mp.descending_partitions[0]) == 1


def test_multiple_features():
    """Partitions computed independently per feature."""
    X = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    mp = MonotonicPartitioner().fit(X, np.array([1, 2, 3]))
    assert len(mp.ascending_partitions) == 2


def test_original_indices_preserved():
    """MMIs contain original sample indices, not sorted positions."""
    mp = MonotonicPartitioner().fit(np.array([[3.0], [1.0], [2.0]]), np.array([3, 1, 2]))
    np.testing.assert_array_equal(mp.ascending_partitions[0][0], [1, 2, 0])


def test_fit_returns_self():
    mp = MonotonicPartitioner()
    assert mp.fit(np.array([[1.0], [2.0]]), np.array([1, 2])) is mp
