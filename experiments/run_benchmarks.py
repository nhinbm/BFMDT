#!/usr/bin/env python
"""Run BFMDT benchmarks on datasets from the paper.

Usage:
    python experiments/run_benchmarks.py                                # breast-wisconsin only
    python experiments/run_benchmarks.py --datasets breast-wisconsin wine
    python experiments/run_benchmarks.py --datasets all                 # all registered datasets
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import StratifiedKFold
from datasets.loader import load_dataset, list_available_datasets
from bfmdt import BFMDTClassifier
from bfmdt.metrics import Evaluator


def run_single_dataset(name, n_folds=5, random_seed=42, sigma="auto",
                       delta=0.01, max_reducts=50):
    """Run stratified k-fold CV on a single dataset.

    Args:
        name (str): Dataset name recognized by load_dataset().
        n_folds (int): Number of CV folds.
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

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_cas = []
    fold_maes = []
    start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = BFMDTClassifier(sigma=sigma, delta=delta, max_reducts=max_reducts)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ca = Evaluator.classification_accuracy(y_test, y_pred)
        mae = Evaluator.mean_absolute_error_ordinal(y_test, y_pred)
        fold_cas.append(ca)
        fold_maes.append(mae)
        print(f"  Fold {fold_idx}/{n_folds}: CA={ca:.4f}, MAE={mae:.4f}")

    elapsed = time.time() - start
    mean_ca = np.mean(fold_cas)
    std_ca = np.std(fold_cas)
    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)

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
    parser = argparse.ArgumentParser(description="BFMDT Benchmark Runner")
    parser.add_argument(
        "--datasets", nargs="+", default=["breast-wisconsin"],
        help="Dataset names to test (default: breast-wisconsin). Use 'all' for all.",
    )
    parser.add_argument("--sigma", type=str, default="auto")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--max-reducts", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.datasets == ["all"]:
        dataset_names = list_available_datasets()
    else:
        dataset_names = args.datasets

    sigma = args.sigma if args.sigma == "auto" else float(args.sigma)

    print(f"Config: sigma={sigma}, delta={args.delta}, max_reducts={args.max_reducts}, "
          f"n_folds={args.n_folds}, seed={args.seed}")

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

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Dataset':<25} {'CA':>10} {'MAE':>10} {'Time':>8}")
        print("-" * 55)
        for r in results:
            print(f"{r['dataset']:<25} {r['mean_ca']:>10.4f} {r['mean_mae']:>10.4f} "
                  f"{r['elapsed']:>7.1f}s")


if __name__ == "__main__":
    main()
