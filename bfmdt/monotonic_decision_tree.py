import numpy as np


class MonotonicDecisionTree:
    """Build a binary decision tree using Rank Mutual Information as split criterion.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision values of shape (n_samples,).
        feature_indices (list[int]): Which feature columns to use for splitting.
        direction (str): 'ascending' (ARMI) or 'descending' (DRMI).
        delta (float): Minimum RMI threshold to split. Defaults to 0.01.

    Returns:
        dsl (np.ndarray): Decision Support Level matrix of shape (n_samples, n_classes). Values in [0, 1].
    """

    def __init__(self, direction="ascending", delta=0.01, n_classes=None):
        self.direction = direction
        self.delta = delta
        self.n_classes = n_classes
        self.root = None

    def fit(self, X, y, feature_indices=None):
        # TODO: implement tree building with ARMI/DRMI split criterion
        raise NotImplementedError

    def predict_dsl(self, X):
        # TODO: implement traversal, return DSL matrix (n_samples, n_classes)
        raise NotImplementedError
