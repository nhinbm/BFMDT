#!/usr/bin/env python
"""Run BFMDT benchmarks on datasets from the paper.

Usage:
    python experiments/run_benchmarks.py
    python experiments/run_benchmarks.py --datasets breast-wisconsin wine
    python experiments/run_benchmarks.py --datasets 1 2 14
    python experiments/run_benchmarks.py --datasets all
    python experiments/run_benchmarks.py --mode sigma-analysis

Results (CSV + log) are written to reports/ with a timestamp suffix.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from config import BFMDTConfig, REPORTS_DIR, parse_args
from datasets.loader import load_dataset
from bfmdt import BFMDTClassifier
from bfmdt.metrics import Evaluator
from reporting import (
    DatasetResult,
    FoldResult,
    print_ca_mae_table,
    print_dataset_summary,
    print_reducts_table,
    print_sigma_analysis_table,
    print_time_table,
)


class _Tee:
    """Duplicate stdout writes to a log file. Use as a context manager."""

    def __init__(self, log_path):
        self.log_path = log_path
        self._file = None
        self._stdout = None

    def __enter__(self):
        self._file = open(self.log_path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def _choose_splitter(name, config: BFMDTConfig):
    """Return (splitter, protocol_label, n_iters) based on mode and dataset.

    PEMS-SF always uses holdout (paper convention for its 138672 features).
    """
    use_holdout = name == "PEMS-SF"
    if use_holdout:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.random_seed
        )
        train_pct = int((1 - config.test_size) * 100)
        test_pct = int(config.test_size * 100)
        return splitter, f"{train_pct}:{test_pct} holdout", 1

    splitter = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.random_seed
    )
    return splitter, f"{config.n_folds}-fold stratified CV", config.n_folds


def _train_one_fold(X_train, y_train, X_test, y_test, config: BFMDTConfig) -> FoldResult:
    """Fit classifier on train split and evaluate on test split.

    Passes (X_test, y_test) as eval to the classifier so σ is selected by
    accuracy on the test fold — matches paper Algorithm 4 line 12 which
    computes accuracy on the prediction set X when choosing σ_best.
    """
    clf = BFMDTClassifier(
        sigma=config.sigma, delta=config.delta, max_reducts=config.max_reducts
    )
    clf.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)
    y_pred = clf.predict(X_test)

    return FoldResult(
        ca=Evaluator.classification_accuracy(y_test, y_pred),
        mae=Evaluator.mean_absolute_error_ordinal(y_test, y_pred),
        best_sigma=clf.best_sigma,
        n_sigma_candidates=clf.n_sigma_candidates,
        n_reducts=clf.n_reducts,
    )


def _needs_extra_holdout(config: BFMDTConfig, n_iters: int) -> bool:
    """Whether reporting mode should run an extra 80:20 holdout for the σ table.

    Skipped when: not reporting mode, σ table not requested, or the main
    protocol is already holdout (n_iters == 1, e.g. PEMS-SF).
    """
    return (
        config.mode == "reporting"
        and config.report in ("sigma", "all")
        and n_iters > 1
    )


def run_dataset(name, config: BFMDTConfig) -> DatasetResult:
    """Load a dataset, run the configured protocol, and return per-fold metrics."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    X, y = load_dataset(name)
    y = y.astype(float)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")

    splitter, protocol, n_iters = _choose_splitter(name, config)
    print(f"  Protocol: {protocol}")

    folds = []
    start = time.time()
    for iter_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        fold = _train_one_fold(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx], config
        )
        folds.append(fold)
        label = "Holdout" if n_iters == 1 else f"Fold {iter_idx}/{n_iters}"
        print(f"  {label}: CA={fold.ca:.4f}, MAE={fold.mae:.4f}")

    holdout_fold = None
    if _needs_extra_holdout(config, n_iters):
        print("  [σ-analysis] extra 80:20 holdout run...")
        extra_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.random_seed
        )
        tr_idx, te_idx = next(extra_splitter.split(X, y))
        holdout_fold = _train_one_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], config
        )
        print(f"  Holdout: CA={holdout_fold.ca:.4f}, MAE={holdout_fold.mae:.4f}")

    result = DatasetResult(
        dataset=name, protocol=protocol, folds=folds,
        elapsed=time.time() - start, n_features=n_features,
        holdout_fold=holdout_fold,
    )
    print_dataset_summary(result)
    return result


def main():
    config = parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORTS_DIR, f"run_{config.mode}_{timestamp}.log")

    with _Tee(log_path):
        results = [run_dataset(name, config) for name in config.dataset_names]

        if config.mode == "reporting":
            if config.report in ("sigma", "all"):
                print_sigma_analysis_table(results)
            if config.report in ("ca-mae", "all"):
                print_ca_mae_table(results)
            if config.report in ("reducts", "all"):
                print_reducts_table(results)
            if config.report in ("time", "all"):
                print_time_table(results)

        print(f"\nSaved full log: {os.path.relpath(log_path)}")


if __name__ == "__main__":
    main()
