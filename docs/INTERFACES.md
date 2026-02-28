# BFMDT Interfaces

Mapped to each step in [PIPELINE.md](./PIPELINE.md).

---

## Training Step 1 -- Preprocessing

```python
class Preprocessor:
    """Clean and normalize raw data for the pipeline.

    Args:
        X (np.ndarray): Raw feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Raw decision labels of shape (n_samples,).

    Returns:
        X_clean (np.ndarray): Normalized feature matrix in [0, 1], missing values imputed. Shape (n_samples, n_features).
        y_clean (np.ndarray): Ordinal integer labels, samples with missing decisions removed. Shape (n_samples,).
    """
```

---

## Training Step 2 -- Monotonic Partition (Algorithm 1)

```python
class MonotonicPartitioner:
    """Partition samples into Maximal Monotonic Intervals (MMIs) per feature.

    Args:
        X (np.ndarray): Normalized feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Normalized decision values of shape (n_samples,).

    Returns:
        ascending_partitions (list[list[np.ndarray]]): AMP per feature. Each feature has a list of MMIs, each MMI is an array of sample indices.
        descending_partitions (list[list[np.ndarray]]): DMP per feature. Same structure.
    """
```

---

## Training Step 3 -- Fitting Degree Matrix (Algorithm 2)

```python
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
```

---

## Training Step 4 -- Sigma Candidate Selection

```python
class SigmaSelector:
    """Find threshold candidates from the fitting degree matrix.

    Args:
        fitting_matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features).
        min_candidates (int): Minimum number of sigma candidates. Defaults to 5.
        max_candidates (int): Maximum number of sigma candidates. Defaults to 30.

    Returns:
        sigmas (list[float]): List of sigma candidate values.
    """
```

---

## Training Step 5 -- Feature Selection (Algorithm 3)

```python
class FeatureSelector:
    """Find minimal feature subsets (reducts) that preserve monotonic information.

    Args:
        fitting_matrix (np.ndarray): Fitting degree matrix of shape (n_samples, n_features).
        sigma (float): Threshold for binarizing the fitting degree matrix.
        max_reducts (int): Maximum number of reducts to return. Defaults to 50.

    Returns:
        reducts (list[list[int]]): List of reducts. Each reduct is a list of feature indices.
    """
```

---

## Training Step 6 -- Build Trees

```python
class MonotonicDecisionTree:
    """Build a binary decision tree using Rank Mutual Information as split criterion.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision values of shape (n_samples,).
        feature_indices (list[int]): Which feature columns to use for splitting.
        direction (str): 'ascending' (ARMI) or 'descending' (DRMI).
        delta (float): Minimum RMI threshold to split. Defaults to 0.01.

    Returns:
        dsl (np.ndarray): Decision Support Level matrix of shape (n_samples, n_classes). Values in [0, 1].
    """
```

---

## Training Step 7-8 + Prediction Step 1-4 -- Main Classifier

```python
class BFMDTClassifier:
    """Orchestrate the full BFMDT pipeline.

    Args:
        sigma (float | str): Fitting degree threshold, or 'auto' for automatic selection. Defaults to 'auto'.
        delta (float): RMI threshold for tree splitting. Defaults to 0.01.
        max_reducts (int): Maximum feature subsets per sigma. Defaults to 50.

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
```

---

## Evaluation

```python
class Evaluator:
    """Compute classification performance metrics for ordinal classification.

    Args:
        y_true (np.ndarray): True labels of shape (n_samples,).
        y_pred (np.ndarray): Predicted labels of shape (n_samples,).

    Returns:
        accuracy (float): Classification accuracy (CA). Range [0, 1].
        mae (float): Mean absolute error using ordinal class distance (MAE). Range [0, K-1].
    """
```

---

## Dataset Loading

```python
class DatasetLoader:
    """Load benchmark datasets used in the paper.

    Args:
        name (str): Dataset name string, or None to load all 18 datasets.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Decision labels of shape (n_samples,).
        n_classes (int): Number of decision classes.
    """
```
