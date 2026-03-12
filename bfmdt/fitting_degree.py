import numpy as np

from .utils import compute_inversion_count_for_feature


class FittingDegreeComputer:
    """Measure how well each sample fits the monotonic relationship under each feature.

    Args:
        X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Normalized decision values of shape (n_samples,).
        ascending_partitions (list[list[np.ndarray]]): AMP from MonotonicPartitioner.
        descending_partitions (list[list[np.ndarray]]): DMP from MonotonicPartitioner.

    Returns:
        matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features). Values in [0, 1].
        monotone_directions (np.ndarray): Direction per feature of shape (n_features,). +1 (increasing) or -1 (decreasing).
        X_adjusted (np.ndarray): Feature matrix with decreasing features inverted. Shape (n_samples, n_features).
    """

    def __init__(self):
        self.matrix = None
        self.monotone_directions = None
        self.X_adjusted = None

    def fit(self, X, y, ascending_partitions, descending_partitions):
        """Compute the fitting degree matrix (Algorithm 2).

        Args:
            X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Normalized decision values of shape (n_samples,).
            ascending_partitions (list[list[np.ndarray]]): AMP from MonotonicPartitioner.
            descending_partitions (list[list[np.ndarray]]): DMP from MonotonicPartitioner.

        Returns:
            self: Fitted computer with matrix, monotone_directions,
                and X_adjusted populated.
        """
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("X must be normalized to [0, 1]")

        n_samples, n_features = X.shape

        matrix = np.zeros((n_samples, n_features))
        monotone_directions = np.zeros(n_features, dtype=int)
        X_adjusted = X.copy()

        # Precompute class sizes: |[x_i]_D| for each class
        classes, class_counts = np.unique(y, return_counts=True)
        class_size = dict(zip(classes, class_counts))

        # Global factor denominator: N(N-1)
        denom = n_samples * (n_samples - 1) if n_samples > 1 else 1

        for j in range(n_features):
            F_asc = FittingDegreeComputer._compute_fitting_for_direction(
                X, y, j, ascending_partitions, class_size, n_samples, denom
            )
            F_desc = FittingDegreeComputer._compute_fitting_for_direction(
                X, y, j, descending_partitions, class_size, n_samples, denom
            )

            # --- Direction decision (Eq. 22) ---
            if np.sum(F_asc) >= np.sum(F_desc):
                monotone_directions[j] = 1
                matrix[:, j] = F_asc
            else:
                monotone_directions[j] = -1
                matrix[:, j] = F_desc

        # Invert decreasing features (Algorithm 2, lines 13-15)
        desc_mask = monotone_directions == -1
        X_adjusted[:, desc_mask] = 1.0 - X_adjusted[:, desc_mask]

        self.matrix = matrix
        self.monotone_directions = monotone_directions
        self.X_adjusted = X_adjusted
        return self

    @staticmethod
    def _compute_fitting_for_direction(
        X, y, j, partitions, class_size, n_samples, denom
    ):
        """Compute per-sample fitting degree for one feature in one direction.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Decision values of shape (n_samples,).
            j (int): Feature index.
            partitions (list[list[np.ndarray]]): AMP or DMP from MonotonicPartitioner.
            class_size (dict): Mapping from class label to total count.
            n_samples (int): Total number of samples.
            denom (int): Global factor denominator N(N-1).

        Returns:
            F (np.ndarray): Fitting degree per sample, shape (n_samples,).
        """
        inv = compute_inversion_count_for_feature(X, y, j)
        global_factor = 1.0 - 2.0 * inv / denom
        return FittingDegreeComputer._compute_local_fitting(
            partitions[j], y, class_size, n_samples, global_factor
        )

    @staticmethod
    def _compute_local_fitting(partition, y, class_size, n_samples, global_factor):
        """Compute per-sample fitting degree for one feature direction.

        Args:
            partition (list[np.ndarray]): MMIs for one feature (AMP or DMP).
            y (np.ndarray): Decision values of shape (n_samples,).
            class_size (dict): Mapping from class label to total count.
            n_samples (int): Total number of samples.
            global_factor (float): Global monotonicity factor for this feature.

        Returns:
            F (np.ndarray): Fitting degree per sample, shape (n_samples,).
        """
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
