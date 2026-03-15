import pytest
import numpy as np
from bfmdt.monotonic_decision_tree import MonotonicDecisionTree


@pytest.fixture
def paper_dataset():
    """
    Dataset from Table I of the BFMDT paper.
    Features: a(0), b(1), c(2), d(3), e(4), f(5), g(6), h(7).
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


# ---------------------------------------------------------------------------
# Example 5 / Fig 3 -- Reduct {b} (feature index 1)
# ---------------------------------------------------------------------------

class TestARMIReductB:
    """ARMI tree for reduct {b}. Paper Fig 3 top-left."""

    def test_tree_structure(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="ascending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[1])

        # Root: split on b <= 0.3
        root = tree.root
        assert root.is_leaf is False
        assert root.feature_index == 1
        assert root.split_value == 0.3

        # Left child: split on b <= 0.1
        assert root.left.is_leaf is False
        assert root.left.feature_index == 1
        assert root.left.split_value == 0.1

        # Left-left leaf: {x1, x3, x2} all class 1
        assert root.left.left.is_leaf is True
        np.testing.assert_array_equal(root.left.left.dsl, [1.0, 0.0, 0.0])

        # Left-right leaf: {x5, x4} all class 2
        assert root.left.right.is_leaf is True
        np.testing.assert_array_equal(root.left.right.dsl, [0.0, 1.0, 0.0])

        # Right child: split on b <= 0.5
        assert root.right.is_leaf is False
        assert root.right.feature_index == 1
        assert root.right.split_value == 0.5

        # Right-left leaf: {x6, x7} classes 2, 3
        assert root.right.left.is_leaf is True
        np.testing.assert_array_equal(root.right.left.dsl, [0.0, 0.5, 0.5])

        # Right-right leaf: {x8, x9} all class 3
        assert root.right.right.is_leaf is True
        np.testing.assert_array_equal(root.right.right.dsl, [0.0, 0.0, 1.0])

    def test_dsl_predictions(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="ascending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[1])
        dsl = tree.predict_dsl(X)

        expected = np.array([
            [1.0, 0.0, 0.0],  # x1: b=0.0 -> leaf [1,0,0]
            [1.0, 0.0, 0.0],  # x2: b=0.1 -> leaf [1,0,0]
            [1.0, 0.0, 0.0],  # x3: b=0.0 -> leaf [1,0,0]
            [0.0, 1.0, 0.0],  # x4: b=0.3 -> leaf [0,1,0]
            [0.0, 1.0, 0.0],  # x5: b=0.2 -> leaf [0,1,0]
            [0.0, 0.5, 0.5],  # x6: b=0.5 -> leaf [0,0.5,0.5]
            [0.0, 0.5, 0.5],  # x7: b=0.5 -> leaf [0,0.5,0.5]
            [0.0, 0.0, 1.0],  # x8: b=0.6 -> leaf [0,0,1]
            [0.0, 0.0, 1.0],  # x9: b=1.0 -> leaf [0,0,1]
        ])
        np.testing.assert_allclose(dsl, expected)
        np.testing.assert_allclose(dsl.sum(axis=1), np.ones(9))


class TestDRMIReductB:
    """DRMI tree for reduct {b}. Paper Fig 3 top-right."""

    def test_tree_structure(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="descending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[1])

        # Root: split on b <= 0.1
        root = tree.root
        assert root.is_leaf is False
        assert root.feature_index == 1
        assert root.split_value == 0.1

        # Left leaf: {x1, x3, x2} all class 1
        assert root.left.is_leaf is True
        np.testing.assert_array_equal(root.left.dsl, [1.0, 0.0, 0.0])

        # Right child: split on b <= 0.3
        assert root.right.is_leaf is False
        assert root.right.feature_index == 1
        assert root.right.split_value == 0.3

        # Right-left leaf: {x5, x4} all class 2
        assert root.right.left.is_leaf is True
        np.testing.assert_array_equal(root.right.left.dsl, [0.0, 1.0, 0.0])

        # Right-right: split on b <= 0.5
        assert root.right.right.is_leaf is False
        assert root.right.right.feature_index == 1
        assert root.right.right.split_value == 0.5

        # Right-right-left leaf: {x6, x7} classes 2, 3
        assert root.right.right.left.is_leaf is True
        np.testing.assert_array_equal(root.right.right.left.dsl,
                                      [0.0, 0.5, 0.5])

        # Right-right-right leaf: {x8, x9} all class 3
        assert root.right.right.right.is_leaf is True
        np.testing.assert_array_equal(root.right.right.right.dsl,
                                      [0.0, 0.0, 1.0])

    def test_dsl_predictions(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="descending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[1])
        dsl = tree.predict_dsl(X)

        expected = np.array([
            [1.0, 0.0, 0.0],  # x1: b=0.0 -> leaf [1,0,0]
            [1.0, 0.0, 0.0],  # x2: b=0.1 -> leaf [1,0,0]
            [1.0, 0.0, 0.0],  # x3: b=0.0 -> leaf [1,0,0]
            [0.0, 1.0, 0.0],  # x4: b=0.3 -> leaf [0,1,0]
            [0.0, 1.0, 0.0],  # x5: b=0.2 -> leaf [0,1,0]
            [0.0, 0.5, 0.5],  # x6: b=0.5 -> leaf [0,0.5,0.5]
            [0.0, 0.5, 0.5],  # x7: b=0.5 -> leaf [0,0.5,0.5]
            [0.0, 0.0, 1.0],  # x8: b=0.6 -> leaf [0,0,1]
            [0.0, 0.0, 1.0],  # x9: b=1.0 -> leaf [0,0,1]
        ])
        np.testing.assert_allclose(dsl, expected)
        np.testing.assert_allclose(dsl.sum(axis=1), np.ones(9))


# ---------------------------------------------------------------------------
# Example 5 / Fig 3 -- Reduct {d, f} (feature indices [3, 5])
# ---------------------------------------------------------------------------

class TestARMIReductDF:
    """ARMI tree for reduct {d, f}. Paper Fig 3 bottom-left."""

    def test_tree_structure(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="ascending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[3, 5])

        # Root: split on f <= 0.4
        root = tree.root
        assert root.is_leaf is False
        assert root.feature_index == 5  # f
        assert root.split_value == 0.4

        # Left child: split on d <= 0.2
        assert root.left.feature_index == 3  # d
        assert root.left.split_value == 0.2

        # Left-left leaf: {x1, x2} all class 1
        assert root.left.left.is_leaf is True
        np.testing.assert_array_equal(root.left.left.dsl, [1.0, 0.0, 0.0])

        # Left-right child: split on f <= 0.0
        assert root.left.right.is_leaf is False
        assert root.left.right.feature_index == 5  # f
        assert root.left.right.split_value == 0.0

        # Left-right-left leaf: {x3} class 1
        assert root.left.right.left.is_leaf is True
        np.testing.assert_array_equal(root.left.right.left.dsl,
                                      [1.0, 0.0, 0.0])

        # Left-right-right leaf: {x4, x5} all class 2
        assert root.left.right.right.is_leaf is True
        np.testing.assert_array_equal(root.left.right.right.dsl,
                                      [0.0, 1.0, 0.0])

        # Right child: split on f <= 0.5
        assert root.right.feature_index == 5  # f
        assert root.right.split_value == 0.5

        # Right-left leaf: {x6, x7} classes 2, 3
        assert root.right.left.is_leaf is True
        np.testing.assert_array_equal(root.right.left.dsl, [0.0, 0.5, 0.5])

        # Right-right leaf: {x8, x9} all class 3
        assert root.right.right.is_leaf is True
        np.testing.assert_array_equal(root.right.right.dsl, [0.0, 0.0, 1.0])

    def test_dsl_predictions(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="ascending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[3, 5])
        dsl = tree.predict_dsl(X)

        expected = np.array([
            [1.0, 0.0, 0.0],  # x1: f=0.4<=0.4, d=0.2<=0.2 -> [1,0,0]
            [1.0, 0.0, 0.0],  # x2: f=0.2<=0.4, d=0.0<=0.2 -> [1,0,0]
            [1.0, 0.0, 0.0],  # x3: f=0.0<=0.4, d=0.5>0.2, f=0.0<=0.0 -> [1,0,0]
            [0.0, 1.0, 0.0],  # x4: f=0.1<=0.4, d=0.5>0.2, f=0.1>0.0 -> [0,1,0]
            [0.0, 1.0, 0.0],  # x5: f=0.4<=0.4, d=0.7>0.2, f=0.4>0.0 -> [0,1,0]
            [0.0, 0.5, 0.5],  # x6: f=0.5>0.4, f=0.5<=0.5 -> [0,0.5,0.5]
            [0.0, 0.5, 0.5],  # x7: f=0.5>0.4, f=0.5<=0.5 -> [0,0.5,0.5]
            [0.0, 0.0, 1.0],  # x8: f=0.7>0.4, f=0.7>0.5 -> [0,0,1]
            [0.0, 0.0, 1.0],  # x9: f=1.0>0.4, f=1.0>0.5 -> [0,0,1]
        ])
        np.testing.assert_allclose(dsl, expected)
        np.testing.assert_allclose(dsl.sum(axis=1), np.ones(9))


class TestDRMIReductDF:
    """DRMI tree for reduct {d, f}. Paper Fig 3 bottom-right."""

    def test_tree_structure(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="descending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[3, 5])

        # Root: split on d <= 0.5
        root = tree.root
        assert root.is_leaf is False
        assert root.feature_index == 3  # d
        assert root.split_value == 0.5

        # Left child: split on d <= 0.2
        assert root.left.feature_index == 3  # d
        assert root.left.split_value == 0.2

        # Left-left leaf: {x1, x2} all class 1
        assert root.left.left.is_leaf is True
        np.testing.assert_array_equal(root.left.left.dsl, [1.0, 0.0, 0.0])

        # Left-right child: split on f <= 0.0
        assert root.left.right.feature_index == 5  # f
        assert root.left.right.split_value == 0.0

        # Left-right-left leaf: {x3} class 1
        assert root.left.right.left.is_leaf is True
        np.testing.assert_array_equal(root.left.right.left.dsl,
                                      [1.0, 0.0, 0.0])

        # Left-right-right leaf: {x4} class 2
        assert root.left.right.right.is_leaf is True
        np.testing.assert_array_equal(root.left.right.right.dsl,
                                      [0.0, 1.0, 0.0])

        # Right child: split on f <= 0.5
        assert root.right.feature_index == 5  # f
        assert root.right.split_value == 0.5

        # Right-left child: split on d <= 0.7
        assert root.right.left.feature_index == 3  # d
        assert root.right.left.split_value == 0.7

        # Right-left-left leaf: {x5} class 2
        assert root.right.left.left.is_leaf is True
        np.testing.assert_array_equal(root.right.left.left.dsl,
                                      [0.0, 1.0, 0.0])

        # Right-left-right leaf: {x6, x7} classes 2, 3
        assert root.right.left.right.is_leaf is True
        np.testing.assert_array_equal(root.right.left.right.dsl,
                                      [0.0, 0.5, 0.5])

        # Right-right leaf: {x8, x9} all class 3
        assert root.right.right.is_leaf is True
        np.testing.assert_array_equal(root.right.right.dsl, [0.0, 0.0, 1.0])

    def test_dsl_predictions(self, paper_dataset):
        X, y = paper_dataset
        tree = MonotonicDecisionTree(direction="descending", delta=0.001,
                                     )
        tree.fit(X, y, feature_indices=[3, 5])
        dsl = tree.predict_dsl(X)

        expected = np.array([
            [1.0, 0.0, 0.0],  # x1: d=0.2<=0.5, d=0.2<=0.2 -> [1,0,0]
            [1.0, 0.0, 0.0],  # x2: d=0.0<=0.5, d=0.0<=0.2 -> [1,0,0]
            [1.0, 0.0, 0.0],  # x3: d=0.5<=0.5, d=0.5>0.2, f=0.0<=0.0 -> [1,0,0]
            [0.0, 1.0, 0.0],  # x4: d=0.5<=0.5, d=0.5>0.2, f=0.1>0.0 -> [0,1,0]
            [0.0, 1.0, 0.0],  # x5: d=0.7>0.5, f=0.4<=0.5, d=0.7<=0.7 -> [0,1,0]
            [0.0, 0.5, 0.5],  # x6: d=1.0>0.5, f=0.5<=0.5, d=1.0>0.7 -> [0,0.5,0.5]
            [0.0, 0.5, 0.5],  # x7: d=0.8>0.5, f=0.5<=0.5, d=0.8>0.7 -> [0,0.5,0.5]
            [0.0, 0.0, 1.0],  # x8: d=1.0>0.5, f=0.7>0.5 -> [0,0,1]
            [0.0, 0.0, 1.0],  # x9: d=0.7>0.5, f=1.0>0.5 -> [0,0,1]
        ])
        np.testing.assert_allclose(dsl, expected)
        np.testing.assert_allclose(dsl.sum(axis=1), np.ones(9))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_pure_node_stopping_condition():
    """Tree stops and creates a leaf when all samples have the same class."""
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([1, 1, 1])

    tree = MonotonicDecisionTree()
    tree.fit(X, y)

    assert tree.root.is_leaf is True

    dsl = tree.predict_dsl(X)
    np.testing.assert_array_equal(dsl[0], [1.0])
    np.testing.assert_allclose(dsl.sum(axis=1), np.ones(3))
