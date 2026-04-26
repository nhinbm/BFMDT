import numpy as np


class Node:
    """Data structure for a node in the decision tree."""

    def __init__(self, dsl, is_leaf=False):
        self.dsl = dsl
        self.is_leaf = is_leaf
        self.feature_index = None
        self.split_value = None
        self.left = None
        self.right = None


class MonotonicDecisionTreeV2:
    """Version 2 of the monotonic decision tree.

    Differences from v1, applied because paper Fig 3 / test fixtures appear to
    diverge from the published Eq 13/14/16 formulas:

    1. Split candidates are MIDPOINTS between consecutive distinct values:
       c = (v_t + v_{t+1}) / 2, instead of v1's lower-bound `unique_vals[:-1]`.
       The induced partition is identical (binarisation depends only on which
       side of c each value falls on), but the stored `split_value` differs.

    2. ARMI / DRMI formulas match paper Eq 13/14/16 strictly:
       - ARMI^≤(a_j, c, D) sums over the LEFT branch using the ≤ relation on D
         (Eq 16: term for x_i in R collapses to log(1) = 0).
       - DRMI^≥ is the symmetric dual: sum over the RIGHT branch using ≥ on D.

    3. Caller wiring is consistent with the names:
       - direction='ascending' uses ARMI (LEFT-sum, ≤).
       - direction='descending' uses DRMI (RIGHT-sum, ≥).
       v1's `_compute_armi` actually computed paper's DRMI on the right branch
       and v1's caller passed mask_R into it -- two inversions that cancelled
       out at the tree level but produced score values inconsistent with Eq 16.
    """

    def __init__(self, direction="ascending", delta=0.01, n_classes=None):
        if direction not in ["ascending", "descending"]:
            raise ValueError("direction must be 'ascending' or 'descending'")
        self.direction = direction
        self.delta = delta
        self.n_classes = n_classes
        self.root = None
        self.classes_ = None

    def fit(self, X, y, feature_indices=None):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            if self.n_classes is None:
                self.n_classes = len(self.classes_)

        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))

        self.root = self._build_tree(X, y, feature_indices)
        return self

    def _compute_dsl(self, y):
        """Class-proportion DSL at a node (paper Def 19)."""
        dsl = np.zeros(self.n_classes)
        if len(y) == 0:
            return dsl

        classes, counts = np.unique(y, return_counts=True)
        for c, count in zip(classes, counts):
            idx = np.where(self.classes_ == c)[0]
            if len(idx) > 0:
                dsl[idx[0]] = count / len(y)
        return dsl

    def _compute_armi(self, mask_L, y):
        """ARMI^≤ per paper Eq 16: sum over LEFT branch with ≤ on D."""
        N = len(y)
        L_size = np.sum(mask_L)
        if L_size == 0 or L_size == N:
            return 0.0

        classes, counts = np.unique(y, return_counts=True)
        counts_le = np.cumsum(counts)
        D_map = dict(zip(classes, counts_le))

        y_L = y[mask_L]
        D_i = np.array([D_map[yi] for yi in y_L])

        classes_L, counts_L = np.unique(y_L, return_counts=True)
        counts_L_le = np.cumsum(counts_L)
        L_D_map = dict(zip(classes_L, counts_L_le))
        L_D_i = np.array([L_D_map[yi] for yi in y_L])

        ratio = (L_size * D_i) / (N * L_D_i)
        return -np.sum(np.log(ratio)) / N

    def _compute_drmi(self, mask_R, y):
        """DRMI^≥ symmetric dual of Eq 16: sum over RIGHT branch with ≥ on D."""
        N = len(y)
        R_size = np.sum(mask_R)
        if R_size == 0 or R_size == N:
            return 0.0

        classes, counts = np.unique(y, return_counts=True)
        counts_ge = np.cumsum(counts[::-1])[::-1]
        D_map = dict(zip(classes, counts_ge))

        y_R = y[mask_R]
        D_i = np.array([D_map[yi] for yi in y_R])

        classes_R, counts_R = np.unique(y_R, return_counts=True)
        counts_R_ge = np.cumsum(counts_R[::-1])[::-1]
        R_D_map = dict(zip(classes_R, counts_R_ge))
        R_D_i = np.array([R_D_map[yi] for yi in y_R])

        ratio = (R_size * D_i) / (N * R_D_i)
        return -np.sum(np.log(ratio)) / N

    def _build_tree(self, X, y, feature_indices):
        """Recursively build the Monotonic tree."""
        dsl = self._compute_dsl(y)

        # Stopping #1 (pure node) + #3 (single sample collapses to pure).
        if len(np.unique(y)) <= 1 or len(feature_indices) == 0:
            return Node(dsl, is_leaf=True)

        best_rmi = -np.inf
        best_feature = None
        best_split = None

        for f in feature_indices:
            unique_vals = np.unique(X[:, f])
            # Stopping #2 (feature constant on U_i): skip; if every feature
            # is constant, best_feature stays None and falls into Stopping #4.
            if len(unique_vals) <= 1:
                continue

            # Midpoint candidates between consecutive distinct values.
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for c in midpoints:
                if self.direction == "ascending":
                    mask_L = X[:, f] <= c
                    rmi = self._compute_armi(mask_L, y)
                else:
                    mask_R = X[:, f] > c
                    rmi = self._compute_drmi(mask_R, y)

                if rmi > best_rmi:
                    best_rmi = rmi
                    best_feature = f
                    best_split = c

        # Stopping #4 (no split found, or max RMI < δ).
        if best_feature is None or best_rmi < self.delta:
            return Node(dsl, is_leaf=True)

        mask_L = X[:, best_feature] <= best_split
        mask_R = ~mask_L

        node = Node(dsl, is_leaf=False)
        node.feature_index = best_feature
        node.split_value = best_split

        node.left = self._build_tree(X[mask_L], y[mask_L], feature_indices)
        node.right = self._build_tree(X[mask_R], y[mask_R], feature_indices)

        return node

    def predict_dsl(self, X):
        """Traverse the tree to output DSL predictions for each sample."""
        if self.root is None:
            raise ValueError("Tree is not fitted yet.")

        predictions = []
        for i in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[i, node.feature_index] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.dsl)

        return np.array(predictions)
