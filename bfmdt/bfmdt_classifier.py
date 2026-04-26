import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from .preprocessing import Preprocessor
from .monotonic_partition import MonotonicPartitioner
from .fitting_degree import FittingDegreeComputer
from .fitting_degree_v2 import FittingDegreeComputerV2
from .sigma_selection import SigmaSelector
from .feature_selection import FeatureSelector
from .monotonic_decision_tree import MonotonicDecisionTree
from .monotonic_decision_tree_v2 import MonotonicDecisionTreeV2
from .metrics import Evaluator
from .utils import map_classes_to_ordinal


class BFMDTClassifier(ClassifierMixin, BaseEstimator):
    """Orchestrate the full BFMDT pipeline.

    Args:
        sigma (float | str): Fitting degree threshold, or 'auto' for automatic selection. Defaults to 'auto'.
        delta (float): RMI threshold for tree splitting. Defaults to 0.01.
        max_reducts (int): Maximum feature subsets per sigma. Defaults to 50.
        allow_missing (bool): When False, NaN in features raises during preprocessing
            (contract assertion). When True (default), NaN cells are mean-imputed.
        fitting_version (int): 1 = paper Algorithm 2 as published; 2 = symmetric
            directional-inversion variant in fitting_degree_v2. Defaults to 1.
        tree_version (int): 1 = matches paper Fig 3 / current tests; 2 = strict
            paper Eq 13/14/16 with midpoint splits. Defaults to 1.
        min_sigma_candidates (int): Min sigma candidates passed to SigmaSelector. Defaults to 5.
        max_sigma_candidates (int): Max sigma candidates passed to SigmaSelector. Defaults to 30.
        max_sigma_iterations (int): Hard cap on sp adjustment iterations. Defaults to 1000.

    fit(X, y):
        Args:
            X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Raw decision labels of shape (n_samples,).

    predict(X):
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Returns:
            y_pred (np.ndarray): Predicted class labels of shape (n_samples,).

    predict_proba(X):
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Returns:
            proba (np.ndarray): Fused DSL normalized to probabilities of shape (n_samples, n_classes).
    """

    def __init__(
        self,
        sigma="auto",
        delta=0.01,
        max_reducts=50,
        allow_missing=True,
        fitting_version=1,
        tree_version=1,
        min_sigma_candidates=5,
        max_sigma_candidates=30,
        max_sigma_iterations=1000,
    ):
        self.sigma = sigma
        self.delta = delta
        self.max_reducts = max_reducts
        self.allow_missing = allow_missing
        self.fitting_version = fitting_version
        self.tree_version = tree_version
        self.min_sigma_candidates = min_sigma_candidates
        self.max_sigma_candidates = max_sigma_candidates
        self.max_sigma_iterations = max_sigma_iterations
        self.trees = None
        self.best_sigma = None
        self.n_sigma_candidates = None
        self.n_reducts = None
        self.preprocessor = None
        self.fitting_matrix = None
        self.monotone_directions = None
        self.classes_ = None
        self.n_classes_ = None
        self.label_map_ = None
        self.inverse_label_map_ = None

    def fit(self, X, y, X_eval=None, y_eval=None):
        """Train the BFMDT classifier (Algorithm 4).

        Args:
            X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Raw decision labels of shape (n_samples,).
            X_eval (np.ndarray | None): Optional held-out features for σ selection.
            y_eval (np.ndarray | None): Optional held-out labels for σ selection.

        Note:
            σ scored on (X_eval, y_eval) if given (paper protocol — pass the
            test fold to reproduce Algorithm 4 line 12), else on an internal
            stratified 80/20 holdout with final trees rebuilt on full train.

        Returns:
            self: The fitted classifier.
        """
        # Step 1: Preprocess
        self.preprocessor = Preprocessor(allow_missing=self.allow_missing)
        X_clean, y_clean = self.preprocessor.fit_transform(X, y)

        # Step 2: Map classes to ordinal integers
        y_ordinal, self.label_map_ = map_classes_to_ordinal(y_clean)
        self.inverse_label_map_ = {v: k for k, v in self.label_map_.items()}
        self.classes_ = np.array(sorted(self.label_map_.keys()))
        self.n_classes_ = len(self.label_map_)

        # Step 3: Monotonic Partition (Algorithm 1)
        partitioner = MonotonicPartitioner()
        partitioner.fit(X_clean, y_ordinal)

        # Step 4: Fitting Degree Matrix (Algorithm 2)
        FDClass = FittingDegreeComputerV2 if self.fitting_version == 2 else FittingDegreeComputer
        fd = FDClass()
        fd.fit(X_clean, y_ordinal,
               partitioner.ascending_partitions,
               partitioner.descending_partitions)
        self.fitting_matrix = fd.matrix
        self.monotone_directions = fd.monotone_directions
        X_adjusted = fd.X_adjusted

        # Step 4b: σ scoring set. σ MUST be tuned on a held-out fold; tuning
        # on training accuracy biases σ toward overfit trees. Caller-supplied
        # eval is honoured; otherwise carve a stratified 80/20 holdout.
        if X_eval is not None and y_eval is not None:
            X_eval_clean = self.preprocessor.transform(X_eval)
            eval_X_adj = X_eval_clean.copy()
            desc_mask = self.monotone_directions == -1
            eval_X_adj[:, desc_mask] = 1.0 - eval_X_adj[:, desc_mask]
            eval_y_ord = np.array([self.label_map_[yi] for yi in y_eval])

            inner_train_X, inner_train_y = X_adjusted, y_ordinal
            inner_val_X, inner_val_y = eval_X_adj, eval_y_ord
            using_internal_split = False
        else:
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=0
                )
                train_idx, val_idx = next(splitter.split(X_adjusted, y_ordinal))
            except ValueError:
                splitter = ShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=0
                )
                train_idx, val_idx = next(splitter.split(X_adjusted, y_ordinal))
            inner_train_X = X_adjusted[train_idx]
            inner_train_y = y_ordinal[train_idx]
            inner_val_X = X_adjusted[val_idx]
            inner_val_y = y_ordinal[val_idx]
            using_internal_split = True

        # Step 5: Sigma candidates
        if self.sigma == "auto":
            sigma_candidates = SigmaSelector(
                min_candidates=self.min_sigma_candidates,
                max_candidates=self.max_sigma_candidates,
                max_iterations=self.max_sigma_iterations,
            ).select(self.fitting_matrix)
        else:
            sigma_candidates = [float(self.sigma)]
        self.n_sigma_candidates = len(sigma_candidates)

        # Steps 6-8: For each σ, build trees on inner_train, score on inner_val.
        TreeClass = MonotonicDecisionTreeV2 if self.tree_version == 2 else MonotonicDecisionTree
        best_accuracy = -1.0
        best_inner_trees = []
        best_sigma = None
        best_reducts = None
        fs = FeatureSelector(max_reducts=self.max_reducts)

        for sigma_val in sigma_candidates:
            reducts = fs.find_reducts(self.fitting_matrix, sigma_val)
            if not reducts:
                continue

            # Step 6: Build ARMI + DRMI trees per reduct on inner_train.
            inner_trees = []
            for reduct in reducts:
                for direction in ["ascending", "descending"]:
                    tree = TreeClass(
                        direction=direction,
                        delta=self.delta,
                        n_classes=self.n_classes_,
                    )
                    tree.classes_ = np.arange(self.n_classes_)
                    tree.fit(inner_train_X, inner_train_y, feature_indices=reduct)
                    inner_trees.append(tree)

            # Step 7: Score on inner_val (held-out fold).
            y_pred = self._predict_with_trees(inner_trees, inner_val_X)
            acc = Evaluator.classification_accuracy(inner_val_y, y_pred)

            # Step 8: Track best σ.
            if acc > best_accuracy:
                best_accuracy = acc
                best_inner_trees = inner_trees
                best_sigma = sigma_val
                best_reducts = reducts

        # Step 9: When the inner split chopped 20% off training, rebuild final
        # trees on the FULL training set with the chosen σ's reducts. With a
        # caller-supplied eval, inner_train was already the full set so the
        # trees from Step 6 are reusable.
        if best_sigma is not None and using_internal_split:
            final_trees = []
            for reduct in best_reducts:
                for direction in ["ascending", "descending"]:
                    tree = TreeClass(
                        direction=direction,
                        delta=self.delta,
                        n_classes=self.n_classes_,
                    )
                    tree.classes_ = np.arange(self.n_classes_)
                    tree.fit(X_adjusted, y_ordinal, feature_indices=reduct)
                    final_trees.append(tree)
            self.trees = final_trees
        else:
            self.trees = best_inner_trees

        self.best_sigma = best_sigma
        self.n_reducts = len(best_reducts) if best_reducts is not None else 0
        return self

    def _predict_with_trees(self, trees, X_adjusted):
        """Fuse DSL from a set of trees on already-adjusted X → ordinal predictions."""
        dsl_sum = np.zeros((X_adjusted.shape[0], self.n_classes_))
        for tree in trees:
            dsl_sum += tree.predict_dsl(X_adjusted)
        return np.argmax(dsl_sum, axis=1)

    def _compute_fused_dsl(self, X):
        """Compute fused DSL for samples (prediction steps 1-3).

        Args:
            X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Fused DSL matrix of shape (n_samples, n_classes).
        """
        if self.trees is None or len(self.trees) == 0:
            raise ValueError("Model has not been fitted yet or no trees were built. Call fit() first.")

        # Step 1: Normalize using training parameters
        X_clean = self.preprocessor.transform(X)

        # Step 2: Invert decreasing features
        X_adjusted = X_clean.copy()
        desc_mask = self.monotone_directions == -1
        X_adjusted[:, desc_mask] = 1.0 - X_adjusted[:, desc_mask]

        # Step 3: Sum DSL from all trees
        dsl_sum = np.zeros((X_adjusted.shape[0], self.n_classes_))
        for tree in self.trees:
            dsl_sum += tree.predict_dsl(X_adjusted)

        return dsl_sum

    def predict(self, X):
        """Predict class labels for samples.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            y_pred (np.ndarray): Predicted class labels of shape (n_samples,).
        """
        dsl = self._compute_fused_dsl(X)
        y_pred_ordinal = np.argmax(dsl, axis=1)
        return np.array([self.inverse_label_map_[i] for i in y_pred_ordinal])

    def predict_proba(self, X):
        """Predict class probabilities via fused DSL.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            proba (np.ndarray): Fused DSL normalized to probabilities of shape (n_samples, n_classes).
        """
        dsl = self._compute_fused_dsl(X)
        row_sums = dsl.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return dsl / row_sums
