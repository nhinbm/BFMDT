import pytest
import numpy as np
from bfmdt.feature_selection import FeatureSelector

@pytest.fixture
def example_2_matrix():
    """
    Fitting Degree Matrix (Table II) from the BFMDT paper.
    Columns represent features: a, b, c, d, e, f, g, h (indices 0 to 7).
    """
    return np.array([
        [0.23, 1.0, 0.24, 0.94, 0.35, 0.63, 0.41, 0.19],  # x1
        [0.46, 1.0, 0.48, 0.94, 0.18, 0.63, 0.20, 0.19],  # x2
        [0.46, 1.0, 0.48, 0.94, 0.35, 0.31, 0.41, 0.09],  # x3
        [0.23, 1.0, 0.24, 0.63, 0.53, 0.31, 0.20, 0.09],  # x4
        [0.23, 1.0, 0.24, 0.63, 0.53, 0.63, 0.20, 0.19],  # x5
        [0.23, 1.0, 0.24, 0.31, 0.53, 0.63, 0.20, 0.19],  # x6
        [0.23, 1.0, 0.24, 0.63, 0.18, 0.94, 0.20, 0.28],  # x7
        [0.46, 1.0, 0.48, 0.31, 0.35, 0.94, 0.20, 0.28],  # x8
        [0.46, 1.0, 0.48, 0.63, 0.35, 0.94, 0.20, 0.28],  # x9
    ])

def test_absorb_matrix():
    """
    Test the _absorb_matrix function which removes supersets according to the absorption law.
    If row i is a subset of row j, row j will be deleted.
    """
    selector = FeatureSelector()
    
    # Sample binary matrix
    M = np.array([
        [1, 0, 1], # {0, 2}
        [1, 1, 1], # {0, 1, 2} - Superset of row 0, will be deleted
        [0, 1, 0], # {1}
        [0, 1, 1]  # {1, 2} - Superset of row 2, will be deleted
    ])
    
    M_absorbed = selector._absorb_matrix(M)
    
    expected = np.array([
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    assert M_absorbed.shape == (2, 3), "The absorbed matrix must have 2 rows remaining"
    np.testing.assert_array_equal(M_absorbed, expected)

def test_find_reducts_example_paper(example_2_matrix):
    """
    Test Algorithm 3 with data from Examples 2, 3, and 4 of the paper.
    """
    selector = FeatureSelector(max_reducts=50)
    sigma = 0.5
    
    reducts = selector.find_reducts(example_2_matrix, sigma)
    
    # According to Example 4 in the paper, the reducts include {b} and {d, f}
    # With 0-indexing:
    # b -> index 1
    # d -> index 3
    # f -> index 5
    
    assert len(reducts) > 0, "The algorithm must find at least 1 reduct"
    
    expected_reducts = [[1], [3, 5]]
    assert len(reducts) == len(expected_reducts), f"Expected {len(expected_reducts)} reducts, got {len(reducts)}"
    for r in expected_reducts:
        assert sorted(r) in [sorted(x) for x in reducts], f"Expected reduct {r} not found in {reducts}"

def test_edge_case_all_zeros():
    """
    Test the edge case where the fitting degree matrix has no values >= sigma.
    """
    selector = FeatureSelector()
    fitting_matrix = np.zeros((5, 4))
    sigma = 0.5
    
    reducts = selector.find_reducts(fitting_matrix, sigma)
    
    assert reducts == [], "There should be no reducts if there are no fitting degree values >= sigma"

def test_edge_case_all_ones():
    """
    Test the edge case with an all-ones matrix (all features are strongly related).
    Note: Step 4 of Algorithm 3 removes "all 1" rows, so no reducts are found via this heuristic.
    """
    selector = FeatureSelector()
    fitting_matrix = np.ones((5, 4))
    sigma = 0.5
    
    reducts = selector.find_reducts(fitting_matrix, sigma)
    
    assert reducts == [], "Rows containing all 1s are removed, thus the returned list is empty"

def test_max_reducts_limit():
    """
    Test if the max_reducts parameter works correctly.
    """
    selector = FeatureSelector(max_reducts=1)
    # Symmetric matrix to generate multiple OFS feature branches
    fitting_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    sigma = 0.5

    reducts = selector.find_reducts(fitting_matrix, sigma)

    assert len(reducts) == 1, "Only the specified maximum number of reducts should be returned (max_reducts=1)"


def test_option_c_tie_break_picks_non_ofs():
    """When the inner greedy ties between an OFS feature and a non-OFS feature,
    Option C must pick the non-OFS one to avoid producing a non-minimal reduct.

    Fixture (after binarize at sigma=0.5):
        Row 0: {a, b}    -> OFS = {a, b} (cols 0, 1)
        Row 1: {a, c}

    Branch a_k = b (col 1):
        Delete rows where col 1 = 1 -> row 0 gone, leaving row 1 = {a, c}.
        Greedy ties: col_sum(a) = col_sum(c) = 1.
        Without Option C (plain `tied[0]`): pick a -> reduct = {a, b}, which is
            non-minimal because {a} is already a reduct from the a-seed branch.
        With Option C: a is in OFS, c is not -> pick c -> reduct = {b, c}, minimal.
    """
    matrix = np.array([
        [0.8, 0.7, 0.2],
        [0.6, 0.3, 0.9],
    ])

    reducts = FeatureSelector(max_reducts=50).find_reducts(matrix, 0.5)

    sorted_reducts = {tuple(sorted(r)) for r in reducts}
    assert sorted_reducts == {(0,), (1, 2)}, (
        f"Option C should yield {{a}}, {{b, c}}; got {reducts}. A regression "
        f"that drops the non-OFS preference would yield {{a}}, {{a, b}}."
    )


def test_reducts_cover_constraint_rows(example_2_matrix):
    """Algorithm 3 invariant: every returned reduct must cover all constraint rows.

    A constraint row is a row of M_sigma that is not trivially all-0 or all-1
    (those are removed in step 4). For each reduct R, every constraint row must
    have at least one column in R set to 1 -- otherwise R fails to satisfy the
    monotonic related family.
    """
    sigma = 0.5
    reducts = FeatureSelector(max_reducts=50).find_reducts(example_2_matrix, sigma)
    assert len(reducts) > 0

    M_sigma = (example_2_matrix >= sigma).astype(int)
    n_features = M_sigma.shape[1]
    row_sums = M_sigma.sum(axis=1)
    constraint_rows = M_sigma[(row_sums > 0) & (row_sums < n_features)]

    for r in reducts:
        coverage = constraint_rows[:, r].sum(axis=1)
        uncovered = np.where(coverage == 0)[0]
        assert len(uncovered) == 0, (
            f"Reduct {r} fails to cover constraint rows {uncovered.tolist()}"
        )
