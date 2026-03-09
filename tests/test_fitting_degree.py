import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner
from bfmdt.fitting_degree import FittingDegreeComputer


@pytest.fixture
def mixed_features():
    """Two features: first mostly ascending, second mostly descending."""
    X = np.array([[0.1, 0.9], [0.5, 0.8], [0.3, 0.4], [0.8, 0.1]])
    y = np.array([0, 1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputer().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    return fd, X, y


def test_matrix(mixed_features):
    fd, X, y = mixed_features
    expected = np.array([
        [0.8333, 1.0],
        [0.8333, 1.0],
        [0.8333, 1.0],
        [0.8333, 1.0],
    ])
    np.testing.assert_allclose(fd.matrix, expected, atol=0.001)


def test_directions(mixed_features):
    fd, X, y = mixed_features
    np.testing.assert_array_equal(fd.monotone_directions, [1, -1])


def test_x_adjusted(mixed_features):
    fd, X, y = mixed_features
    expected = np.array([
        [0.1, 0.1],
        [0.5, 0.2],
        [0.3, 0.6],
        [0.8, 0.9],
    ])
    np.testing.assert_allclose(fd.X_adjusted, expected)


def test_fit_returns_self(mixed_features):
    fd, X, y = mixed_features
    mp = MonotonicPartitioner().fit(X, y)
    fd2 = FittingDegreeComputer()
    assert fd2.fit(X, y, mp.ascending_partitions, mp.descending_partitions) is fd2
