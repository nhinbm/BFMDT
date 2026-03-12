import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner
from bfmdt.fitting_degree import FittingDegreeComputer


@pytest.fixture
def paper_example():
    """Table 1 from the paper: 9 samples, 8 features, 3 classes."""
    X = np.array([
        [0.1, 0.0, 0.3, 0.2, 1.0, 0.4, 0.6, 0.8],
        [0.3, 0.1, 0.6, 0.0, 0.6, 0.2, 0.1, 0.9],
        [0.3, 0.0, 0.5, 0.5, 0.9, 0.0, 0.7, 0.6],
        [0.1, 0.3, 0.3, 0.5, 0.1, 0.1, 0.8, 0.7],
        [0.3, 0.2, 0.7, 0.7, 0.0, 0.4, 0.4, 0.5],
        [0.0, 0.5, 0.1, 1.0, 0.1, 0.5, 0.1, 0.3],
        [0.6, 0.5, 0.1, 0.8, 0.6, 0.5, 0.0, 0.2],
        [0.1, 0.6, 1.0, 1.0, 0.2, 0.7, 0.3, 0.2],
        [0.1, 1.0, 1.0, 0.7, 0.1, 1.0, 1.0, 0.0],
    ])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputer().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    return fd, X, y


def test_matrix(paper_example):
    fd, X, y = paper_example
    expected = np.array([
        [0.23, 1.0, 0.24, 0.94, 0.35, 0.63, 0.41, 0.19],
        [0.46, 1.0, 0.48, 0.94, 0.18, 0.63, 0.20, 0.19],
        [0.46, 1.0, 0.48, 0.94, 0.35, 0.31, 0.41, 0.09],
        [0.23, 1.0, 0.24, 0.63, 0.53, 0.31, 0.20, 0.09],
        [0.23, 1.0, 0.24, 0.63, 0.53, 0.63, 0.20, 0.19],
        [0.23, 1.0, 0.24, 0.31, 0.53, 0.63, 0.20, 0.19],
        [0.23, 1.0, 0.24, 0.63, 0.18, 0.94, 0.20, 0.28],
        [0.46, 1.0, 0.48, 0.31, 0.35, 0.94, 0.20, 0.28],
        [0.46, 1.0, 0.48, 0.63, 0.35, 0.94, 0.20, 0.28],
    ])
    np.testing.assert_allclose(fd.matrix, expected, atol=0.01)


def test_fit_returns_self(paper_example):
    fd, X, y = paper_example
    mp = MonotonicPartitioner().fit(X, y)
    fd2 = FittingDegreeComputer()
    assert fd2.fit(X, y, mp.ascending_partitions, mp.descending_partitions) is fd2
