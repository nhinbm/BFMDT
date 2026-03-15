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


class MonotonicDecisionTree:
    """Build a binary decision tree using Rank Mutual Information as split criterion.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision values of shape (n_samples,).
        feature_indices (list[int]): Which feature columns to use for splitting.
        direction (str): 'ascending' (ARMI) or 'descending' (DRMI).
        delta (float): Minimum RMI threshold to split. Defaults to 0.01.
        n_classes (int): Number of decision classes. Defaults to None (inferred from y).

    Returns:
        dsl (np.ndarray): Decision Support Level matrix of shape (n_samples, n_classes). Values in [0, 1].
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
        # Identify all classes to standardize the output size of the DSL array
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            if self.n_classes is None:
                self.n_classes = len(self.classes_)
                
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
            
        self.root = self._build_tree(X, y, feature_indices)
        return self

    def _compute_dsl(self, y):
        """Calculate the Decision Support Level (DSL) for the current node."""
        dsl = np.zeros(self.n_classes)
        if len(y) == 0:
            return dsl
            
        classes, counts = np.unique(y, return_counts=True)
        for c, count in zip(classes, counts):
            idx = np.where(self.classes_ == c)[0]
            if len(idx) > 0:
                dsl[idx[0]] = count / len(y)
        return dsl

    def _compute_armi(self, mask_R, y):
        """Calculate Ascending Rank Mutual Information"""
        N = len(y)
        R_size = np.sum(mask_R)
        if R_size == 0 or R_size == N:
            return 0.0
            
        # Calculate the size of the dominance set for y (D_i) over the entire node
        classes, counts = np.unique(y, return_counts=True)
        counts_ge = np.cumsum(counts[::-1])[::-1]
        D_map = dict(zip(classes, counts_ge))
        
        # Map D_i for samples falling into the right branch
        y_R = y[mask_R]
        D_i = np.array([D_map[yi] for yi in y_R])
        
        # Calculate the size of the intersection of the right branch and the dominance set (R_D_i)
        classes_R, counts_R = np.unique(y_R, return_counts=True)
        counts_R_ge = np.cumsum(counts_R[::-1])[::-1]
        R_D_map = dict(zip(classes_R, counts_R_ge))
        R_D_i = np.array([R_D_map[yi] for yi in y_R])
        
        # Calculate ARMI
        ratio = (R_size * D_i) / (N * R_D_i)
        armi = -np.sum(np.log(ratio)) / N
        return armi

    def _compute_drmi(self, mask_L, y):
        """Calculate Descending Rank Mutual Information"""
        N = len(y)
        L_size = np.sum(mask_L)
        if L_size == 0 or L_size == N:
            return 0.0
            
        # Calculate the size of the inferiority set for y (E_i) over the entire node
        classes, counts = np.unique(y, return_counts=True)
        counts_le = np.cumsum(counts)
        E_map = dict(zip(classes, counts_le))
        
        # Map E_i for samples falling into the left branch
        y_L = y[mask_L]
        E_i = np.array([E_map[yi] for yi in y_L])
        
        # Calculate the intersection size of the left branch and the inferiority set (L_E_i)
        classes_L, counts_L = np.unique(y_L, return_counts=True)
        counts_L_le = np.cumsum(counts_L)
        L_E_map = dict(zip(classes_L, counts_L_le))
        L_E_i = np.array([L_E_map[yi] for yi in y_L])
        
        # Calculate DRMI
        ratio = (L_size * E_i) / (N * L_E_i)
        drmi = -np.sum(np.log(ratio)) / N
        return drmi

    def _build_tree(self, X, y, feature_indices):
        """Recursively build the Monotonic tree."""
        dsl = self._compute_dsl(y)
        
        # Stopping condition: Pure node or no features left
        if len(np.unique(y)) <= 1 or len(feature_indices) == 0:
            return Node(dsl, is_leaf=True)
            
        best_rmi = -1
        best_feature = None
        best_split = None
        
        # Iterate through features to find the best split point
        for f in feature_indices:
            unique_vals = np.unique(X[:, f])
            if len(unique_vals) <= 1:
                continue
                
            splits = unique_vals[:-1]
            
            for c in splits:
                if self.direction == "ascending":
                    mask_R = X[:, f] > c
                    rmi = self._compute_armi(mask_R, y)
                else:
                    mask_L = X[:, f] <= c
                    rmi = self._compute_drmi(mask_L, y)
                    
                if rmi > best_rmi:
                    best_rmi = rmi
                    best_feature = f
                    best_split = c
                    
        # Stop splitting if RMI growth is less than the delta threshold
        if best_rmi < self.delta:
            return Node(dsl, is_leaf=True)
            
        # Split the data
        mask_L = X[:, best_feature] <= best_split
        mask_R = ~mask_L
        
        node = Node(dsl, is_leaf=False)
        node.feature_index = best_feature
        node.split_value = best_split
        
        # Recursively build child nodes
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

