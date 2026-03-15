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
    
    expected_reducts = [[1], [3, 1]]
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