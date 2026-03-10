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
        # 1: initialize the feature subsets RED_sigma(A) = empty
        reducts = []

        # 2: set c_ij as 1 if c_ij >= sigma the others as 0
        # denote the related family as M_sigma
        M_sigma = (fitting_matrix >= sigma).astype(int)

        # 3: sort the rows in ascending by each row sum
        row_sums = M_sigma.sum(axis=1)
        sorted_indices = np.argsort(row_sums)
        M_sorted = M_sigma[sorted_indices]

        # 4: remove the rows that are all 0 or all 1
        n_features = M_sorted.shape[1]
        valid_rows = (M_sorted.sum(axis=1) > 0) & (M_sorted.sum(axis=1) < n_features)
        M_filtered = M_sorted[valid_rows]

        if M_filtered.shape[0] == 0:
            return []

        # 5: absorb M_sigma to get the simplified related family M*
        M_star = self._absorb_matrix(M_filtered)

        if M_star.shape[0] == 0:
            return []

        # 6: let OFS = {a_k in A | e*_1k = 1}
        # Get features (column indices) that have a 1 in the first row
        OFS = np.where(M_star[0] == 1)[0].tolist()

        # 7: for a_k in OFS do
        for a_k in OFS:
            # Control condition to limit max subsets
            if len(reducts) >= self.max_reducts:
                break

            # 8: red = {a_k}
            red = [a_k]

            # Create a working copy of M* for this iteration
            M_temp = M_star.copy()
            available_features = list(range(n_features))

            # update M*: delete the rows satisfied e*_ik = 1
            rows_to_keep = M_temp[:, a_k] == 0
            M_temp = M_temp[rows_to_keep]
            
            # update m': logically delete the kth column
            available_features.remove(a_k)

            # 9-16: while n' != 0 and m' != 0 do
            while M_temp.shape[0] > 0 and len(available_features) > 0:
                # = argmax: Greedily pick the feature that covers the most remaining rows
                col_sums = M_temp[:, available_features].sum(axis=0)

                # Break if no remaining feature covers any remaining row
                if np.max(col_sums) == 0:
                    break

                best_idx_relative = np.argmax(col_sums)
                best_feature = available_features[best_idx_relative]

                # red <- a_k (add feature to reduct)
                red.append(best_feature)

                # update M*: delete the rows satisfied by the selected feature
                rows_to_keep = M_temp[:, best_feature] == 0
                M_temp = M_temp[rows_to_keep]

                # update m': delete the selected column
                available_features.remove(best_feature)

            # 17: RED_sigma(A) <- red
            reducts.append(red)

        # 19: return RED_sigma(A)
        return reducts

    def _absorb_matrix(self, M):
        """
        Helper function for Step 5: Absorb operation (Law of Absorption).
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