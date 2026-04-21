#!/usr/bin/env python
"""Run BFMDT benchmarks on datasets from the paper.

Usage:
    python experiments/run_benchmarks.py
    python experiments/run_benchmarks.py --datasets breast-wisconsin wine
    python experiments/run_benchmarks.py --datasets 1 2 14
    python experiments/run_benchmarks.py --datasets all
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from config import BFMDTConfig, parse_args
from datasets.loader import load_dataset
from bfmdt import BFMDTClassifier
from bfmdt.metrics import Evaluator


def run_single_dataset(name, config: BFMDTConfig):
    """Run the paper's evaluation protocol on a single dataset.

    PEMS-SF uses an 80:20 stratified holdout (per the paper, due to its
    138672-feature dimensionality); every other dataset uses stratified k-fold CV.

    Args:
        name (str): Dataset name recognized by load_dataset().
        config (BFMDTConfig): Centralized configuration with classifier and
            experiment hyperparameters.

    Returns:
        dict: Results with keys 'dataset', 'mean_ca', 'std_ca',
              'mean_mae', 'std_mae', 'fold_cas', 'fold_maes', 'elapsed'.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    X, y = load_dataset(name)
    y = y.astype(float)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")

    if name == 'PEMS-SF':
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.random_seed
        )
        protocol = f"{int((1 - config.test_size) * 100)}:{int(config.test_size * 100)} holdout"
        n_iters = 1
    else:
        splitter = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.random_seed
        )
        protocol = f"{config.n_folds}-fold stratified CV"
        n_iters = config.n_folds
    print(f"  Protocol: {protocol}")

    fold_cas = []
    fold_maes = []
    start = time.time()

    for iter_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = BFMDTClassifier(
            sigma=config.sigma, delta=config.delta, max_reducts=config.max_reducts
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ca = Evaluator.classification_accuracy(y_test, y_pred)
        mae = Evaluator.mean_absolute_error_ordinal(y_test, y_pred)
        fold_cas.append(ca)
        fold_maes.append(mae)
        label = "Holdout" if n_iters == 1 else f"Fold {iter_idx}/{n_iters}"
        print(f"  {label}: CA={ca:.4f}, MAE={mae:.4f}")

    elapsed = time.time() - start
    mean_ca = np.mean(fold_cas)
    std_ca = np.std(fold_cas)
    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)

    if n_iters == 1:
        print(f"\n  CA:  {mean_ca:.4f}")
        print(f"  MAE: {mean_mae:.4f}")
    else:
        print(f"\n  Mean CA:  {mean_ca:.4f} +/- {std_ca:.4f}")
        print(f"  Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "dataset": name,
        "mean_ca": mean_ca,
        "std_ca": std_ca,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "fold_cas": fold_cas,
        "fold_maes": fold_maes,
        "elapsed": elapsed,
    }


def main():
    config = parse_args()

    results = []
    for name in config.dataset_names:
        result = run_single_dataset(name, config)
        results.append(result)


if __name__ == "__main__":
    main()
