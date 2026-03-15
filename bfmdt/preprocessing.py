import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
        self.label_encoders_ = {} 
        self.categorical_cols_ = []

    def fit_transform(self, X, y):
        valid_y_mask = ~np.isnan(y)
        X_clean = X[valid_y_mask].copy()
        y_clean = y[valid_y_mask].copy()
        
        n_samples, n_features = X_clean.shape
        self.categorical_cols_ = []
        
        for col_idx in range(n_features):
            col_data = X_clean[:, col_idx]
            non_nan_mask = ~pd.isna(col_data)
            
            if non_nan_mask.sum() > 0:
                try:
                    col_data[non_nan_mask].astype(float)
                except ValueError:
                    self.categorical_cols_.append(col_idx)
                    le = LabelEncoder()
                    
                    vals_to_encode = col_data[non_nan_mask].astype(str)
                    encoded_vals = le.fit_transform(vals_to_encode)
                    
                    X_clean[non_nan_mask, col_idx] = encoded_vals
                    
                    self.label_encoders_[col_idx] = le

        X_clean = X_clean.astype(float)

        self.feature_mean_ = np.nanmean(X_clean, axis=0)
        self.feature_mean_[np.isnan(self.feature_mean_)] = 0.0 
        
        nan_mask = np.isnan(X_clean)
        X_clean = np.where(nan_mask, self.feature_mean_, X_clean)

        self.feature_min_ = np.min(X_clean, axis=0)
        self.feature_max_ = np.max(X_clean, axis=0)
        feature_range = self.feature_max_ - self.feature_min_
        
        self.valid_features_mask_ = feature_range > 0
        
        X_clean = X_clean[:, self.valid_features_mask_]
        self.feature_min_ = self.feature_min_[self.valid_features_mask_]
        self.feature_max_ = self.feature_max_[self.valid_features_mask_]
        self.feature_mean_ = self.feature_mean_[self.valid_features_mask_]
        feature_range = feature_range[self.valid_features_mask_]
        
        X_clean = (X_clean - self.feature_min_) / feature_range
        
        return X_clean, y_clean

    def transform(self, X):
        X_clean = X.copy()
        
        for col_idx in self.categorical_cols_:
            if col_idx < X_clean.shape[1]:
                col_data = X_clean[:, col_idx]
                non_nan_mask = ~pd.isna(col_data)
                
                if non_nan_mask.sum() > 0:
                    le = self.label_encoders_[col_idx]
                    known_classes = set(le.classes_)
                    
                    safe_data = [
                        val if val in known_classes else le.classes_[0] 
                        for val in col_data[non_nan_mask].astype(str)
                    ]
                    
                    X_clean[non_nan_mask, col_idx] = le.transform(safe_data)

        X_clean = X_clean.astype(float)

        X_clean = X_clean[:, self.valid_features_mask_]

        nan_mask = np.isnan(X_clean)
        X_clean = np.where(nan_mask, self.feature_mean_, X_clean)

        feature_range = self.feature_max_ - self.feature_min_
        X_clean = (X_clean - self.feature_min_) / feature_range
        
        X_clean = np.clip(X_clean, 0.0, 1.0)
        
        return X_clean
