import pytest
import numpy as np
from bfmdt.monotonic_decision_tree import MonotonicDecisionTree

@pytest.fixture
def paper_dataset():
    """
    Dataset from Table I of the BFMDT paper.
    Features: a, b, c, d, e, f, g, h.
    Classes: 1, 2, 3.
    """
    X = np.array([
        [0.1, 0.0, 0.3, 0.2, 1.0, 0.4, 0.6, 0.8],  # x1
        [0.3, 0.1, 0.6, 0.0, 0.6, 0.2, 0.1, 0.9],  # x2
        [0.3, 0.0, 0.5, 0.5, 0.9, 0.0, 0.7, 0.6],  # x3
        [0.1, 0.3, 0.3, 0.5, 0.1, 0.1, 0.8, 0.7],  # x4
        [0.3, 0.2, 0.7, 0.7, 0.0, 0.4, 0.4, 0.5],  # x5
        [0.0, 0.5, 0.1, 1.0, 0.1, 0.5, 0.1, 0.3],  # x6
        [0.6, 0.5, 0.1, 0.8, 0.6, 0.5, 0.0, 0.2],  # x7
        [0.1, 0.6, 1.0, 1.0, 0.2, 0.7, 0.3, 0.2],  # x8
        [0.1, 1.0, 1.0, 0.7, 0.1, 1.0, 1.0, 0.0],  # x9
    ])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    return X, y

def test_armi_fit_predict(paper_dataset):
    """
    Test Ascending Rank Mutual Information (ARMI) tree building and DSL prediction.
    """
    X, y = paper_dataset
    
    # Initialize tree for ascending direction
    tree = MonotonicDecisionTree(direction="ascending", delta=0.001)
    tree.fit(X, y)
    
    dsl = tree.predict_dsl(X)
    
    # Verify outputs
    assert dsl.shape == (X.shape[0], 3), "DSL matrix should have shape (n_samples, n_classes)"
    
    # The sum of support levels for each sample across all classes must be exactly 1.0
    row_sums = np.sum(dsl, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(X.shape[0]), err_msg="Probabilities in DSL must sum to 1")

def test_drmi_fit_predict_with_subset(paper_dataset):
    """
    Test Descending Rank Mutual Information (DRMI) on a specific feature subset.
    Similar to Example 5 in the paper using reduct {d, f}.
    """
    X, y = paper_dataset
    
    # Feature indices: d is 3, f is 5
    subset_indices = [3, 5]
    
    tree = MonotonicDecisionTree(direction="descending", delta=0.001)
    tree.fit(X, y, feature_indices=subset_indices)
    
    dsl = tree.predict_dsl(X)
    
    assert dsl.shape == (X.shape[0], 3)
    row_sums = np.sum(dsl, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(X.shape[0]))

def test_pure_node_stopping_condition():
    """
    Test if the tree correctly stops and creates a leaf when a node is pure (all samples have the same class).
    """
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([1, 1, 1])  # Pure node
    
    tree = MonotonicDecisionTree()
    tree.fit(X, y)
    
    assert tree.root.is_leaf is True, "The root must be a leaf if all samples belong to the same class"
    
    # Class 1 should have 100% support (1.0), others 0
    dsl = tree.predict_dsl(X)
    assert dsl[0, 0] == 1.0, "DSL for the pure class should be 1.0"
