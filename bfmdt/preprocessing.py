import numpy as np

class Preprocessor:
    """Clean and normalize raw data for the pipeline.

    Args:
        X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Raw decision labels of shape (n_samples,).

    Returns:
        X_clean (np.ndarray): Normalized feature matrix in [0, 1], missing values imputed. Shape (n_samples, n_features).
        y_clean (np.ndarray): Ordinal integer labels, samples with missing decisions removed. Shape (n_samples,).
    """

    def __init__(self):
        self.feature_min_ = None
        self.feature_max_ = None
        self.feature_mean_ = None
        self.valid_features_mask_ = None

    def fit_transform(self, X, y):
        valid_y_mask = ~np.isnan(y)
        X_clean = X[valid_y_mask].copy()
        y_clean = y[valid_y_mask].copy()

        self.feature_mean_ = np.nanmean(X_clean, axis=0)
        self.feature_mean_[np.isnan(self.feature_mean_)] = 0.0 
        
        nan_mask = np.isnan(X_clean)
        # for i in range(X_clean.shape[1]):
        #     X_clean[nan_mask[:, i], i] = self.feature_mean_[i]
        X_clean = np.where(nan_mask, self.feature_mean_, X_clean)

        self.feature_min_ = np.min(X_clean, axis=0)
        self.feature_max_ = np.max(X_clean, axis=0)
        feature_range = self.feature_max_ - self.feature_min_
        
        self.valid_features_mask_ = feature_range > 0
        
        X_clean = X_clean[:, self.valid_features_mask_]
        self.feature_min_ = self.feature_min_[self.valid_features_mask_]
        feature_range = feature_range[self.valid_features_mask_]
        
        X_clean = (X_clean - self.feature_min_) / feature_range
        
        return X_clean, y_clean

    def transform(self, X):
        X_clean = X.copy()
        
        nan_mask = np.isnan(X_clean)
        # for i in range(X_clean.shape[1]):
        #     X_clean[nan_mask[:, i], i] = self.feature_mean_[i]
        X_clean = np.where(nan_mask, self.feature_mean_, X_clean)
        
        X_clean = X_clean[:, self.valid_features_mask_]
        
        feature_range = self.feature_max_[self.valid_features_mask_] - self.feature_min_
        X_clean = (X_clean - self.feature_min_) / feature_range
        
        X_clean = np.clip(X_clean, 0.0, 1.0)
        
        return X_clean