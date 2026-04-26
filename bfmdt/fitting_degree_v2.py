import numpy as np

from .utils import compute_inversion_count_for_feature, count_tied_pairs


class FittingDegreeComputerV2:
    """Version 2 of the fitting degree computation.

    Differs from v1 only in the global monotonicity factor (gf):

      v1:
        gf_asc  = 1 - 2 * inv / (n * (n-1))
        gf_desc = 2 * inv / (n * (n-1))
      where inv = #{ (i,k) : x_i < x_k, y_i > y_k } (one count, asymmetric formulas)

      v2:
        gf_asc  = 1 - 2 * inv_asc  / (n * (n-1))
        gf_desc = 1 - 2 * inv_desc / (n * (n-1))
      where
        inv_asc  = #{ (i,k) : x_i < x_k, y_i > y_k }   (cost for ascending hypothesis)
        inv_desc = #{ (i,k) : x_i < x_k, y_i < y_k }   (cost for descending hypothesis)
      i.e. one inversion count per direction, same symmetric formula.

    Local monotonicity (per-MMI fitting) and direction selection are unchanged.

    Args:
        X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Normalized decision values of shape (n_samples,).
        ascending_partitions (list[list[np.ndarray]]): AMP from MonotonicPartitioner.
        descending_partitions (list[list[np.ndarray]]): DMP from MonotonicPartitioner.

    Returns:
        matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features). Values in [0, 1].
        monotone_directions (np.ndarray): +1 (increasing) or -1 (decreasing) per feature.
        X_adjusted (np.ndarray): X with decreasing features inverted (1 - x).
    """

    def __init__(self):
        self.matrix = None
        self.monotone_directions = None
        self.X_adjusted = None

    def fit(self, X, y, ascending_partitions, descending_partitions):
        """Compute the fitting degree matrix using the v2 global factor."""
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("X must be normalized to [0, 1]")

        n_samples, n_features = X.shape

        matrix = np.zeros((n_samples, n_features))
        monotone_directions = np.zeros(n_features, dtype=int)
        X_adjusted = X.copy()

        classes, class_counts = np.unique(y, return_counts=True)
        class_size = dict(zip(classes, class_counts))

        denom = n_samples * (n_samples - 1) if n_samples > 1 else 1
        total_pairs = n_samples * (n_samples - 1) // 2

        # tied_y is independent of feature; compute once.
        _, y_counts = np.unique(y, return_counts=True)
        tied_y = int(np.sum(y_counts * (y_counts - 1) // 2))

        for j in range(n_features):
            inv_asc, inv_desc = self._compute_directional_inversions(
                X, y, j, total_pairs, tied_y
            )

            gf_asc = 1.0 - 2.0 * inv_asc / denom
            gf_desc = 1.0 - 2.0 * inv_desc / denom

            F_asc = self._compute_local_fitting(
                ascending_partitions[j], y, class_size, n_samples, gf_asc
            )
            F_desc = self._compute_local_fitting(
                descending_partitions[j], y, class_size, n_samples, gf_desc
            )

            if np.sum(F_asc) >= np.sum(F_desc):
                monotone_directions[j] = 1
                matrix[:, j] = F_asc
            else:
                monotone_directions[j] = -1
                matrix[:, j] = F_desc

        desc_mask = monotone_directions == -1
        X_adjusted[:, desc_mask] = 1.0 - X_adjusted[:, desc_mask]

        self.matrix = matrix
        self.monotone_directions = monotone_directions
        self.X_adjusted = X_adjusted
        return self

    @staticmethod
    def _compute_directional_inversions(X, y, j, total_pairs, tied_y):
        """Return (inv_asc, inv_desc) for feature j in O(n log n).

        Derived from the partition identity over all unordered pairs:
            total_pairs = tied_x + tied_y - tied_xy + inv_asc + inv_desc
        so once inv_asc is known via merge-sort, inv_desc is one subtraction away.
        """
        inv_asc = compute_inversion_count_for_feature(X, y, j)
        tied_x = count_tied_pairs(X, j)

        # tied_xy: pairs that are tied in BOTH x_j and y.
        combined = np.stack([X[:, j], y], axis=1)
        _, xy_counts = np.unique(combined, axis=0, return_counts=True)
        tied_xy = int(np.sum(xy_counts * (xy_counts - 1) // 2))

        inv_desc = total_pairs - tied_x - tied_y + tied_xy - inv_asc
        return inv_asc, inv_desc

    @staticmethod
    def _compute_local_fitting(partition, y, class_size, n_samples, global_factor):
        """Per-sample local fitting degree (identical to v1)."""
        F = np.zeros(n_samples)

        for mmi in partition:
            mmi_y = y[mmi]
            unique_classes, inverse, counts = np.unique(
                mmi_y, return_inverse=True, return_counts=True
            )

            same_class_counts = counts[inverse]
            class_sizes_for_mmi = np.array(
                [class_size[c] for c in unique_classes]
            )
            local_parts = same_class_counts / class_sizes_for_mmi[inverse]
            F[mmi] = local_parts * global_factor

        return F
