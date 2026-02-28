import numpy as np


class SigmaSelector:
    """Find threshold candidates from the fitting degree matrix.

    Args:
        fitting_matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features).
        min_candidates (int): Minimum number of sigma candidates. Defaults to 5.
        max_candidates (int): Maximum number of sigma candidates. Defaults to 30.

    Returns:
        sigmas (list[float]): List of sigma candidate values.
    """

    def __init__(self, min_candidates=5, max_candidates=30):
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates

    def select(self, fitting_matrix):
        # TODO: implement
        raise NotImplementedError
