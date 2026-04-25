import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Preprocessor(BaseEstimator, TransformerMixin):
    """Clean and normalize raw feature matrix for the BFMDT pipeline.

    Categorical object columns are encoded via a dict mapping; partially-numeric
    object columns are coerced with non-numeric entries turned into NaN. Missing
    values are imputed column-wise by the mean. Features are min-max scaled to
    [0, 1]. Constant features (zero variance) are dropped.

    Parameters
    ----------
    nan_strategy : {'mean'}, default='mean'
        Strategy for missing-value imputation.
    scaling : {'minmax'}, default='minmax'
        Feature scaling method.
    handle_unknown : {'fallback', 'error'}, default='fallback'
        Behavior when ``transform`` encounters an unseen categorical value.
        - 'fallback': map to index 0.
        - 'error': raise ValueError.
    drop_constant : bool, default=True
        Drop features with zero variance learned from training data.
    nominal_strategy : {'encode', 'drop'}, default='encode'
        How to handle columns containing string values.
        - 'encode': label-encode pure-categorical columns; coerce mixed
          (numeric + string) columns and impute the string entries via mean.
        - 'drop': drop pure-categorical columns AND mixed columns where the
          number of string values exceeds the number of numeric values
          (cannot be normalized reliably). Mixed columns with a numeric
          majority are still coerced + mean-imputed.
    missing_values : tuple, default=('?',)
        Additional sentinels (beyond NaN) to treat as missing.
    allow_missing : bool, default=True
        Contract for NaN handling. When True, NaN cells are imputed by the
        column mean. When False, the presence of any NaN raises ValueError —
        useful as an assertion that a dataset declared "clean" really is.

    Attributes
    ----------
    feature_min_, feature_max_, feature_mean_ : np.ndarray
        Per-feature statistics learned at fit time.
    valid_features_mask_ : np.ndarray of bool
        Mask of features kept after the constant-feature filter and
        (when ``nominal_strategy='drop'``) the nominal-drop filter.
    categorical_cols_ : list of int
        Indices of columns treated as categorical.
    label_mappings_ : dict of {int: dict}
        ``{col_idx: {string_value: integer_code}}`` for categorical columns.
    nominal_dropped_cols_ : list of int
        Indices of columns dropped because of ``nominal_strategy='drop'``.
        Empty when ``nominal_strategy='encode'``. Kept for debug visibility.
    n_features_in_ : int
        Number of features observed at fit time.
    """

    def __init__(
        self,
        nan_strategy='mean',
        scaling='minmax',
        handle_unknown='fallback',
        drop_constant=True,
        nominal_strategy='encode',
        missing_values=('?',),
        allow_missing=True,
    ):
        self.nan_strategy = nan_strategy
        self.scaling = scaling
        self.handle_unknown = handle_unknown
        self.drop_constant = drop_constant
        self.nominal_strategy = nominal_strategy
        self.missing_values = missing_values
        self.allow_missing = allow_missing

    def fit(self, X, y=None):
        """Learn preprocessing parameters from training data.

        Args:
            X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
            y (np.ndarray, optional): Labels. Samples with NaN labels are excluded from learning.

        Returns:
            self
        """
        self._fit_impl(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Args:
            X (np.ndarray): Raw feature matrix.
            y (np.ndarray, optional): Labels. Samples with NaN labels are dropped.

        Returns:
            If ``y`` is None: X_clean (np.ndarray).
            If ``y`` is provided: (X_clean, y_clean) tuple, with samples for NaN labels dropped.
        """
        X_clean, y_clean = self._fit_impl(X, y)
        if y is None:
            return X_clean
        return X_clean, y_clean

    def transform(self, X):
        """Apply learned preprocessing to new data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features_in_).

        Returns:
            X_clean (np.ndarray): Normalized matrix in [0, 1] of shape (n_samples, n_valid_features).
        """
        check_is_fitted(self, ['feature_min_', 'feature_max_', 'valid_features_mask_'])
        X_arr = self._prepare_input(X)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )

        X_work = self._apply_encoding(X_arr)
        X_float = X_work.astype(float)
        X_float = X_float[:, self.valid_features_mask_]

        nan_mask = np.isnan(X_float)
        if nan_mask.any():
            self._reject_unexpected_missing(int(nan_mask.sum()))
            np.copyto(X_float, self.feature_mean_, where=nan_mask)

        feature_range = self.feature_max_ - self.feature_min_
        safe_range = np.where(feature_range > 0, feature_range, 1.0)
        X_float -= self.feature_min_
        X_float /= safe_range
        np.clip(X_float, 0.0, 1.0, out=X_float)

        return X_float

    # --- Private helpers ---

    def _fit_impl(self, X, y):
        self._validate_params()
        X_arr = self._prepare_input(X)

        if y is not None:
            y_arr = np.asarray(y)
            valid_y_mask = ~pd.isna(y_arr)
            X_arr = X_arr[valid_y_mask]
            y_clean = y_arr[valid_y_mask]
        else:
            y_clean = None

        self.n_features_in_ = X_arr.shape[1]
        self.categorical_cols_ = []
        self.label_mappings_ = {}

        X_work, nominal_drop_mask = self._learn_encoding(X_arr)
        self.nominal_dropped_cols_ = np.where(nominal_drop_mask)[0].tolist()
        X_float = X_work.astype(float)

        nan_mask = np.isnan(X_float)
        if nan_mask.any():
            self._reject_unexpected_missing(int(nan_mask.sum()))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.feature_mean_ = np.nanmean(X_float, axis=0)
            self.feature_mean_[np.isnan(self.feature_mean_)] = 0.0
            np.copyto(X_float, self.feature_mean_, where=nan_mask)
        else:
            self.feature_mean_ = np.zeros(X_float.shape[1])

        self.feature_min_ = np.min(X_float, axis=0)
        self.feature_max_ = np.max(X_float, axis=0)
        feature_range = self.feature_max_ - self.feature_min_

        if self.drop_constant:
            self.valid_features_mask_ = feature_range > 0
        else:
            self.valid_features_mask_ = np.ones(len(feature_range), dtype=bool)
        self.valid_features_mask_ &= ~nominal_drop_mask

        if not self.valid_features_mask_.any():
            raise ValueError(
                "All features were dropped during preprocessing "
                f"(nominal_dropped={self.nominal_dropped_cols_}, "
                f"drop_constant={self.drop_constant}). "
                "Cannot continue training."
            )

        self.feature_min_ = self.feature_min_[self.valid_features_mask_]
        self.feature_max_ = self.feature_max_[self.valid_features_mask_]
        self.feature_mean_ = self.feature_mean_[self.valid_features_mask_]
        feature_range = feature_range[self.valid_features_mask_]
        X_float = X_float[:, self.valid_features_mask_]

        X_float -= self.feature_min_
        safe_range = np.where(feature_range > 0, feature_range, 1.0)
        X_float /= safe_range

        return X_float, y_clean

    def _reject_unexpected_missing(self, n_nan):
        """Raise if NaN appears while ``allow_missing=False`` (contract violation)."""
        if not self.allow_missing:
            raise ValueError(
                f"Found NaN in {n_nan} cells but allow_missing=False. "
                "Either pass allow_missing=True or update DATASETS metadata "
                "(set has_missing_values=True for this dataset)."
            )

    def _validate_params(self):
        if self.nan_strategy not in ('mean',):
            raise ValueError(f"nan_strategy must be 'mean', got {self.nan_strategy!r}")
        if self.scaling not in ('minmax',):
            raise ValueError(f"scaling must be 'minmax', got {self.scaling!r}")
        if self.handle_unknown not in ('fallback', 'error'):
            raise ValueError(
                f"handle_unknown must be 'fallback' or 'error', got {self.handle_unknown!r}"
            )
        if self.nominal_strategy not in ('encode', 'drop'):
            raise ValueError(
                f"nominal_strategy must be 'encode' or 'drop', got {self.nominal_strategy!r}"
            )

    def _prepare_input(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        X_arr = X_arr.copy()
        if X_arr.dtype == object and self.missing_values:
            for sentinel in self.missing_values:
                X_arr[X_arr == sentinel] = np.nan
        return X_arr

    def _learn_encoding(self, X):
        nominal_drop_mask = np.zeros(X.shape[1], dtype=bool)
        if X.dtype != object:
            return X, nominal_drop_mask

        drop_strings = self.nominal_strategy == 'drop'
        X_work = X.copy()
        for col_idx in range(X_work.shape[1]):
            col_data = X_work[:, col_idx]
            coerced = pd.to_numeric(col_data, errors='coerce')

            if np.all(np.isnan(coerced)):
                # Pure-categorical column.
                non_nan_mask = ~pd.isna(col_data)
                if non_nan_mask.sum() == 0:
                    continue
                if drop_strings:
                    nominal_drop_mask[col_idx] = True
                    X_work[:, col_idx] = np.nan
                    continue
                self.categorical_cols_.append(col_idx)
                vals = col_data[non_nan_mask].astype(str)
                unique_vals = sorted(np.unique(vals))
                mapping = {v: i for i, v in enumerate(unique_vals)}
                encoded = np.array([mapping[v] for v in vals])
                X_work[non_nan_mask, col_idx] = encoded
                self.label_mappings_[col_idx] = mapping
            else:
                if drop_strings:
                    original_nan = pd.isna(col_data)
                    coerced_nan = np.isnan(coerced.astype(float))
                    numeric_count = int((~original_nan & ~coerced_nan).sum())
                    string_count = int((~original_nan & coerced_nan).sum())
                    if string_count > numeric_count:
                        nominal_drop_mask[col_idx] = True
                        X_work[:, col_idx] = np.nan
                        continue
                X_work[:, col_idx] = coerced

        return X_work, nominal_drop_mask

    def _apply_encoding(self, X):
        if X.dtype != object:
            return X

        X_work = X.copy()
        for col_idx in range(X_work.shape[1]):
            col_data = X_work[:, col_idx]
            if col_idx in self.label_mappings_:
                non_nan_mask = ~pd.isna(col_data)
                if non_nan_mask.sum() > 0:
                    mapping = self.label_mappings_[col_idx]
                    if self.handle_unknown == 'error':
                        for v in col_data[non_nan_mask]:
                            if str(v) not in mapping:
                                raise ValueError(
                                    f"Unknown category {v!r} in column {col_idx}"
                                )
                    encoded = np.array([
                        mapping.get(str(v), 0)
                        for v in col_data[non_nan_mask]
                    ])
                    X_work[non_nan_mask, col_idx] = encoded
            else:
                X_work[:, col_idx] = pd.to_numeric(col_data, errors='coerce')

        return X_work
