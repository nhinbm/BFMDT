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
        """Select sigma candidates from fitting degree matrix values.

        Frequency-based pruning per Section 5.4.1: candidates are values
        whose frequency exceeds L/sp. sp is adjusted by factors of 1.1
        (increase) or 0.9 (decrease) until the candidate count falls
        within [min_candidates, max_candidates].

        Args:
            fitting_matrix (np.ndarray): Fitting degree matrix of shape
                (n_samples, n_features). Values in [0, 1].

        Returns:
            list[float]: Sorted list of sigma candidate values.
        """
        unique_values, counts = np.unique(fitting_matrix, return_counts=True)

        n_samples, n_features = fitting_matrix.shape
        L = max(n_samples, n_features)

        sp = 1.0
        max_iterations = 1000

        for _ in range(max_iterations):
            threshold = L / sp
            mask = counts > threshold
            n_candidates = mask.sum()

            if self.min_candidates <= n_candidates <= self.max_candidates:
                break
            elif n_candidates < self.min_candidates:
                sp *= 1.1
            else:
                sp *= 0.9

        sigmas = sorted(unique_values[mask].tolist())
        return sigmas
