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


def test_monotone_directions(paper_example):
    fd, _, _ = paper_example
    expected = np.array([1, 1, 1, 1, 1, 1, 1, -1])
    np.testing.assert_array_equal(fd.monotone_directions, expected)


def test_X_adjusted_matches_paper_example(paper_example):
    fd, X, _ = paper_example
    expected = X.copy()
    expected[:, 7] = 1.0 - X[:, 7]
    np.testing.assert_allclose(fd.X_adjusted, expected)


def test_X_adjusted_flips_only_descending_features(paper_example):
    fd, X, _ = paper_example
    asc_mask = fd.monotone_directions == 1
    desc_mask = fd.monotone_directions == -1
    np.testing.assert_allclose(fd.X_adjusted[:, asc_mask], X[:, asc_mask])
    np.testing.assert_allclose(fd.X_adjusted[:, desc_mask], 1.0 - X[:, desc_mask])


def test_matrix_in_unit_interval(paper_example):
    fd, _, _ = paper_example
    assert np.all(fd.matrix >= 0.0)
    assert np.all(fd.matrix <= 1.0)


def test_raises_on_unnormalized_X():
    X = np.array([[0.1], [2.0], [0.5]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    with pytest.raises(ValueError, match="normalized"):
        FittingDegreeComputer().fit(
            X, y, mp.ascending_partitions, mp.descending_partitions
        )


def test_strictly_increasing_feature_is_ascending():
    X = np.array([[0.1], [0.4], [0.7]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputer().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    assert fd.monotone_directions[0] == 1
    np.testing.assert_allclose(fd.X_adjusted, X)
    np.testing.assert_allclose(fd.matrix, [[1.0], [1.0], [1.0]])


def test_strictly_decreasing_feature_is_descending():
    X = np.array([[0.7], [0.4], [0.1]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputer().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    assert fd.monotone_directions[0] == -1
    np.testing.assert_allclose(fd.X_adjusted, 1.0 - X)
