import numpy as np


class Evaluator:
    """Compute classification performance metrics for ordinal classification.

    Args:
        y_true (np.ndarray): True labels of shape (n_samples,).
        y_pred (np.ndarray): Predicted labels of shape (n_samples,).

    Returns:
        accuracy (float): Classification accuracy (CA). Range [0, 1].
        mae (float): Mean absolute error using ordinal class distance (MAE). Range [0, K-1].
    """

    @staticmethod
    def classification_accuracy(y_true, y_pred):
        # TODO: implement
        raise NotImplementedError

    @staticmethod
    def mean_absolute_error_ordinal(y_true, y_pred):
        # TODO: implement
        raise NotImplementedError
