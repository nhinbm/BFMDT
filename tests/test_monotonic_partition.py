import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner


@pytest.fixture
def partitioned():
    """X=[3,1,4,2], y=[2,1,4,3] -- unsorted, has breaks in both directions."""
    X = np.array([[3.0], [1.0], [4.0], [2.0]])
    y = np.array([2, 1, 4, 3])
    return MonotonicPartitioner().fit(X, y)


# --- ascending: sort X asc -> y=[1,3,2,4], break at 3>2 -> AMMIs: [1,3], [0,2] ---


def test_ascending_mmi_count(partitioned):
    assert len(partitioned.ascending_partitions[0]) == 2


def test_ascending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][0], [1, 3])


def test_ascending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][1], [0, 2])


# --- descending: sort X desc -> y=[4,2,3,1], breaks at 4>2 and 3>1 ---


def test_descending_mmi_count(partitioned):
    assert len(partitioned.descending_partitions[0]) == 3


def test_descending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][0], [2])


def test_descending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][1], [0, 3])


def test_descending_third_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][2], [1])


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
