import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner
from bfmdt.fitting_degree import FittingDegreeComputer
from bfmdt.sigma_selection import SigmaSelector


@pytest.fixture
def paper_fitting_matrix():
    """Fitting degree matrix from Table 2 in the paper (9 samples, 8 features)."""
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
    return fd.matrix


def test_paper_example_count_in_range(paper_fitting_matrix):
    """Sigma candidates from paper example should be in [5, 30]."""
    selector = SigmaSelector(min_candidates=5, max_candidates=30)
    sigmas = selector.select(paper_fitting_matrix)
    assert 5 <= len(sigmas) <= 30


def test_paper_example_values_from_matrix(paper_fitting_matrix):
    """All returned sigmas must be actual values from the fitting matrix."""
    selector = SigmaSelector()
    sigmas = selector.select(paper_fitting_matrix)
    unique_vals = set(np.unique(paper_fitting_matrix).tolist())
    for s in sigmas:
        assert s in unique_vals


def test_paper_example_sorted(paper_fitting_matrix):
    """Returned sigmas should be sorted in ascending order."""
    selector = SigmaSelector()
    sigmas = selector.select(paper_fitting_matrix)
    assert sigmas == sorted(sigmas)


def test_returns_list_of_floats(paper_fitting_matrix):
    """Return type should be a list of floats."""
    selector = SigmaSelector()
    sigmas = selector.select(paper_fitting_matrix)
    assert isinstance(sigmas, list)
    for s in sigmas:
        assert isinstance(s, float)


def test_controlled_matrix():
    """Test with a known matrix where we can predict the outcome."""
    # 10 samples, 3 features. L = max(10, 3) = 10, sp=1 → threshold=10.
    # Value 0.5 appears 15 times (> 10) → candidate.
    # Value 0.1 appears 12 times (> 10) → candidate.
    # Value 0.9 appears 3 times (< 10) → not candidate.
    matrix = np.array([
        [0.5, 0.5, 0.1],
        [0.5, 0.1, 0.1],
        [0.5, 0.5, 0.1],
        [0.1, 0.1, 0.1],
        [0.5, 0.5, 0.1],
        [0.5, 0.1, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
    ])
    # 0.5 count: col0=7 + col1=6 + col2=4 = 17
    # 0.1 count: col0=2 + col1=3 + col2=5 = 10
    # 0.9 count: col0=1 + col1=1 + col2=1 = 3
    # L = 10, sp=1, threshold=10 → only 0.5 (17>10) passes.
    # That's 1 candidate, < 5, so sp increases until more pass.
    selector = SigmaSelector(min_candidates=2, max_candidates=30)
    sigmas = selector.select(matrix)
    assert len(sigmas) >= 2
    assert all(v in [0.1, 0.5, 0.9] for v in sigmas)


def test_uniform_matrix():
    """Matrix with a single repeated value."""
    matrix = np.full((10, 5), 0.42)
    selector = SigmaSelector(min_candidates=1, max_candidates=30)
    sigmas = selector.select(matrix)
    assert len(sigmas) >= 1
    assert sigmas[0] == pytest.approx(0.42)
