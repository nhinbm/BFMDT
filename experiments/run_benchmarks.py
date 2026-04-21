#!/usr/bin/env python
"""Run BFMDT benchmarks on datasets from the paper.

Usage:
    python experiments/run_benchmarks.py                               
    python experiments/run_benchmarks.py --datasets breast-wisconsin wine
    python experiments/run_benchmarks.py --datasets all                
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from datasets.loader import load_dataset, list_available_datasets
from bfmdt import BFMDTClassifier
from bfmdt.metrics import Evaluator


def run_single_dataset(name, n_folds=5, random_seed=42, sigma="auto",
                       delta=0.01, max_reducts=50):
    """Run the paper's evaluation protocol on a single dataset.

    PEMS-SF uses an 80:20 stratified holdout (per the paper, due to its
    138672-feature dimensionality); every other dataset uses stratified k-fold CV.

    Args:
        name (str): Dataset name recognized by load_dataset().
        n_folds (int): Number of CV folds (ignored for PEMS-SF).
        random_seed (int): Random seed for reproducibility.
        sigma (float | str): Fitting degree threshold or 'auto'.
        delta (float): RMI threshold.
        max_reducts (int): Max feature subsets per sigma.

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
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
        protocol = "80:20 holdout"
        n_iters = 1
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        protocol = f"{n_folds}-fold stratified CV"
        n_iters = n_folds
    print(f"  Protocol: {protocol}")

    fold_cas = []
    fold_maes = []
    start = time.time()

    for iter_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = BFMDTClassifier(sigma=sigma, delta=delta, max_reducts=max_reducts)
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


def parse_args():
    parser = argparse.ArgumentParser(description="BFMDT Benchmark Runner")
    parser.add_argument(
        "--datasets", nargs="+", default=["breast-wisconsin"],
        help="Dataset names to test (default: breast-wisconsin). Use 'all' for all.",
    )
    parser.add_argument(
        "--sigma", type=str, default="auto",
        help="Fitting degree threshold, or 'auto' to select per dataset (default: auto).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="RMI threshold for attribute reduction (default: 0.01).",
    )
    parser.add_argument(
        "--max-reducts", type=int, default=50,
        help="Maximum number of feature subsets per sigma (default: 50).",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.datasets == ["all"]:
        dataset_names = list_available_datasets()
    else:
        dataset_names = args.datasets

    sigma = args.sigma if args.sigma == "auto" else float(args.sigma)

    results = []
    for name in dataset_names:
        result = run_single_dataset(
            name,
            n_folds=args.n_folds,
            random_seed=args.seed,
            sigma=sigma,
            delta=args.delta,
            max_reducts=args.max_reducts,
        )
        results.append(result)


if __name__ == "__main__":
    main()
