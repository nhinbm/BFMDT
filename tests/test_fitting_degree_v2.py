import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner
from bfmdt.fitting_degree_v2 import FittingDegreeComputerV2


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
    fd = FittingDegreeComputerV2().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    return fd, X, y


def test_matrix(paper_example):
    """v2 matrix on Table 1.

    Differs from paper Table II / v1 because v2 uses F_desc (not F_asc) for descending
    features, per the unified F^≤ interface from Algorithm 2 lines 10+16.
    """
    fd, _, _ = paper_example
    expected = np.array([
        [0.25, 1.0, 0.58, 0.94, 0.81, 0.63, 0.44, 0.65],
        [0.50, 1.0, 0.58, 0.94, 0.81, 0.63, 0.22, 0.65],
        [0.50, 1.0, 0.58, 0.94, 0.81, 0.31, 0.44, 0.32],
        [0.25, 1.0, 0.39, 0.63, 0.54, 0.31, 0.22, 0.32],
        [0.25, 1.0, 0.19, 0.63, 0.27, 0.63, 0.22, 0.65],
        [0.25, 1.0, 0.39, 0.31, 0.54, 0.63, 0.22, 0.65],
        [0.25, 1.0, 0.19, 0.63, 0.54, 0.94, 0.22, 0.97],
        [0.50, 1.0, 0.39, 0.31, 0.54, 0.94, 0.22, 0.97],
        [0.50, 1.0, 0.39, 0.63, 0.27, 0.94, 0.22, 0.97],
    ])
    np.testing.assert_allclose(fd.matrix, expected, atol=0.01)


def test_fit_returns_self(paper_example):
    _, X, y = paper_example
    mp = MonotonicPartitioner().fit(X, y)
    fd2 = FittingDegreeComputerV2()
    assert fd2.fit(X, y, mp.ascending_partitions, mp.descending_partitions) is fd2


def test_monotone_directions(paper_example):
    """v2 marks 5 features (a, c, e, g, h) as descending vs v1's 1 (only h).

    With 9/36 tied-y pairs in Table 1, v2's symmetric `gf_desc = 1 - 2·inv_desc/n(n-1)`
    does not collapse to 0 for ascending features the way v1's `2·inv/n(n-1)` does, so
    `sum(F_desc)` overtakes `sum(F_asc)` on more features.
    """
    fd, _, _ = paper_example
    expected = np.array([-1, 1, -1, 1, -1, 1, -1, -1])
    np.testing.assert_array_equal(fd.monotone_directions, expected)


def test_X_adjusted_flips_only_descending_features(paper_example):
    """X_adjusted columns are 1-x for descending features and unchanged for ascending."""
    fd, X, _ = paper_example
    asc_mask = fd.monotone_directions == 1
    desc_mask = fd.monotone_directions == -1
    np.testing.assert_allclose(fd.X_adjusted[:, asc_mask], X[:, asc_mask])
    np.testing.assert_allclose(fd.X_adjusted[:, desc_mask], 1.0 - X[:, desc_mask])


def test_X_adjusted_flips_v2_descending_columns(paper_example):
    """v2 flips columns 0, 2, 4, 6, 7 (the descending features per test_monotone_directions)."""
    fd, X, _ = paper_example
    expected = X.copy()
    for j in (0, 2, 4, 6, 7):
        expected[:, j] = 1.0 - X[:, j]
    np.testing.assert_allclose(fd.X_adjusted, expected)


def test_matrix_in_unit_interval(paper_example):
    fd, _, _ = paper_example
    assert np.all(fd.matrix >= 0.0)
    assert np.all(fd.matrix <= 1.0)


def test_raises_on_unnormalized_X():
    X = np.array([[0.1], [2.0], [0.5]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    with pytest.raises(ValueError, match="normalized"):
        FittingDegreeComputerV2().fit(
            X, y, mp.ascending_partitions, mp.descending_partitions
        )


def test_strictly_increasing_feature_is_ascending():
    """No discordant pairs -> direction +1, X_adjusted unchanged, matrix all 1.0."""
    X = np.array([[0.1], [0.4], [0.7]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputerV2().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    assert fd.monotone_directions[0] == 1
    np.testing.assert_allclose(fd.X_adjusted, X)
    np.testing.assert_allclose(fd.matrix, [[1.0], [1.0], [1.0]])


def test_strictly_decreasing_feature_is_descending():
    """All pairs concordant for descending -> direction -1, X_adjusted = 1-X, matrix all 1.0."""
    X = np.array([[0.7], [0.4], [0.1]])
    y = np.array([1, 2, 3])
    mp = MonotonicPartitioner().fit(X, y)
    fd = FittingDegreeComputerV2().fit(
        X, y, mp.ascending_partitions, mp.descending_partitions
    )
    assert fd.monotone_directions[0] == -1
    np.testing.assert_allclose(fd.X_adjusted, 1.0 - X)
    np.testing.assert_allclose(fd.matrix, [[1.0], [1.0], [1.0]])
