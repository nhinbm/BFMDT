import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .preprocessing import Preprocessor
from .monotonic_partition import MonotonicPartitioner
from .fitting_degree import FittingDegreeComputer
from .sigma_selection import SigmaSelector
from .feature_selection import FeatureSelector
from .monotonic_decision_tree import MonotonicDecisionTree
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

    def __init__(self, sigma="auto", delta=0.01, max_reducts=50, allow_missing=True):
        self.sigma = sigma
        self.delta = delta
        self.max_reducts = max_reducts
        self.allow_missing = allow_missing
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
            Paper Algorithm 4 selects σ_best by accuracy on the prediction set X
            (i.e. the test/eval set), not on the training set. When X_eval/y_eval
            are passed, this implementation follows the paper: σ is tuned on the
            eval set. This is hyperparameter tuning on held-out data, so reported
            metrics on the same eval set are optimistically biased. When eval is
            not passed, σ falls back to training-accuracy selection (sklearn-style).

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
        fd = FittingDegreeComputer()
        fd.fit(X_clean, y_ordinal,
               partitioner.ascending_partitions,
               partitioner.descending_partitions)
        self.fitting_matrix = fd.matrix
        self.monotone_directions = fd.monotone_directions
        X_adjusted = fd.X_adjusted

        eval_X_adj, eval_y_ord = None, None
        if X_eval is not None and y_eval is not None:
            X_eval_clean = self.preprocessor.transform(X_eval)
            eval_X_adj = X_eval_clean.copy()
            desc_mask = self.monotone_directions == -1
            eval_X_adj[:, desc_mask] = 1.0 - eval_X_adj[:, desc_mask]
            eval_y_ord = np.array([self.label_map_[yi] for yi in y_eval])

        # Step 5: Sigma candidates
        if self.sigma == "auto":
            sigma_candidates = SigmaSelector().select(self.fitting_matrix)
        else:
            sigma_candidates = [float(self.sigma)]
        self.n_sigma_candidates = len(sigma_candidates)

        # Steps 6-8: For each sigma, build trees and select best
        best_accuracy = -1.0
        best_trees = []
        best_sigma = None
        best_n_reducts = 0
        fs = FeatureSelector(max_reducts=self.max_reducts)

        for sigma_val in sigma_candidates:
            reducts = fs.find_reducts(self.fitting_matrix, sigma_val)
            if not reducts:
                continue

            # Step 6: Build ARMI + DRMI trees per reduct
            trees = []
            for reduct in reducts:
                for direction in ["ascending", "descending"]:
                    tree = MonotonicDecisionTree(
                        direction=direction,
                        delta=self.delta,
                        n_classes=self.n_classes_,
                    )
                    tree.classes_ = np.arange(self.n_classes_)
                    tree.fit(X_adjusted, y_ordinal, feature_indices=reduct)
                    trees.append(tree)

            # Step 7: Fuse DSL on eval target (paper Alg 4 line 12: acc on X)
            if eval_X_adj is not None:
                target_X, target_y = eval_X_adj, eval_y_ord
            else:
                target_X, target_y = X_adjusted, y_ordinal
            y_pred = self._predict_with_trees(trees, target_X)
            acc = Evaluator.classification_accuracy(target_y, y_pred)

            # Step 8: Track best sigma
            if acc > best_accuracy:
                best_accuracy = acc
                best_trees = trees
                best_sigma = sigma_val
                best_n_reducts = len(reducts)

        self.trees = best_trees
        self.best_sigma = best_sigma
        self.n_reducts = best_n_reducts
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
