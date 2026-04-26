import numpy as np


class FeatureSelector:
    """Find minimal feature subsets (reducts) that preserve monotonic information.

    Args:
        fitting_matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features).
        sigma (float): Threshold for binarizing the fitting degree matrix.
        max_reducts (int): Maximum number of reducts to return. Defaults to 50.

    Returns:
        reducts (list[list[int]]): List of reducts. Each reduct is a list of feature indices.
    """

    def __init__(self, max_reducts=50):
        self.max_reducts = max_reducts

    def find_reducts(self, fitting_matrix, sigma):
        # Initialize the feature subsets RED_sigma(A) = empty
        reducts = []
        seen = set()

        # Set c_ij as 1 if c_ij >= sigma the others as 0
        M_sigma = (fitting_matrix >= sigma).astype(int)

        # Sort the rows in ascending by each row sum
        row_sums = M_sigma.sum(axis=1)
        sorted_indices = np.argsort(row_sums, kind="stable")
        M_sorted = M_sigma[sorted_indices]

        # Remove the rows that are all 0 or all 1
        n_features = M_sorted.shape[1]
        valid_rows = (M_sorted.sum(axis=1) > 0) & (M_sorted.sum(axis=1) < n_features)
        M_filtered = M_sorted[valid_rows]

        if M_filtered.shape[0] == 0:
            return []

        # Absorb M_sigma to get the simplified related family M*
        M_star = self._absorb_matrix(M_filtered)

        if M_star.shape[0] == 0:
            return []

        # Get features (column indices) that have a 1 in the first row
        OFS = np.where(M_star[0] == 1)[0].tolist()

        for a_k in OFS:
            # Control condition to limit max subsets
            if len(reducts) >= self.max_reducts:
                break

            red = [a_k]

            # Per Algorithm 3 line 7: each OFS iteration starts fresh — copy M*
            # and a full feature set. Only the current a_k leaves the available
            # set (line 10), so other OFS features stay reachable for the inner
            # greedy. Dedup at the end (the `seen` set) collapses any reducts
            # produced by different seeds that converge to the same set.
            M_temp = M_star.copy()
            available_features = list(range(n_features))

            rows_to_keep = M_temp[:, a_k] == 0
            M_temp = M_temp[rows_to_keep]
            available_features.remove(a_k)

            ofs_set = set(OFS)
            while M_temp.shape[0] > 0 and len(available_features) > 0:
                # Greedily pick the feature that covers the most remaining rows
                col_sums = M_temp[:, available_features].sum(axis=0)

                # Break if no remaining feature covers any remaining row
                max_sum = col_sums.max()
                if max_sum == 0:
                    break

                # Tie-break: prefer non-OFS features so a seed=a_k branch does
                # not pull in another OFS feature that another seed already owns,
                # which would produce a non-minimal reduct (e.g. {b, d} when {b}
                # is already a reduct from the b-seed branch).
                tied = np.where(col_sums == max_sum)[0]
                non_ofs_tied = [
                    i for i in tied if available_features[i] not in ofs_set
                ]
                best_idx_relative = non_ofs_tied[0] if non_ofs_tied else tied[0]
                best_feature = available_features[best_idx_relative]

                # Add feature to reduct
                red.append(best_feature)

                # Update M*: delete the rows satisfied by the selected feature
                rows_to_keep = M_temp[:, best_feature] == 0
                M_temp = M_temp[rows_to_keep]

                # Update m': remove the selected feature from the available set
                available_features.remove(best_feature)

            key = tuple(sorted(red))
            if key not in seen:
                seen.add(key)
                reducts.append(red)

        return reducts

    def _absorb_matrix(self, M):
        """
        Helper function for Absorb operation.
        Removes rows that are supersets of other rows to simplify the logic expression.
        """
        n_rows = M.shape[0]
        keep_mask = np.ones(n_rows, dtype=bool)

        for i in range(n_rows):
            if not keep_mask[i]:
                continue
            row_i = M[i]
            for j in range(i + 1, n_rows):
                if keep_mask[j]:
                    row_j = M[j]
                    # Check if row_j is a superset of row_i: (row_j AND row_i) == row_i
                    if np.all((row_j & row_i) == row_i):
                        keep_mask[j] = False

        return M[keep_mask]

