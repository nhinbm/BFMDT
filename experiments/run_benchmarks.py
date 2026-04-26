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
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from config import BFMDTConfig, DATA_DIR, DATASETS, NAME_TO_ID, REPORTS_DIR, parse_args
from datasets.loader import load_dataset
from bfmdt import BFMDTClassifier
from bfmdt.metrics import Evaluator
from bfmdt.preprocessing import Preprocessor
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


def _train_one_fold(X_train, y_train, X_test, y_test, config: BFMDTConfig, allow_missing=False) -> FoldResult:
    """Fit classifier on train split and evaluate on test split.

    Test fold is passed as X_eval/y_eval so σ is selected by accuracy on
    the same fold that gets reported -- matches paper Algorithm 4 line 12
    (intentional σ-into-test leakage to reproduce the paper protocol).
    """
    clf = BFMDTClassifier(
        sigma=config.sigma, delta=config.delta, max_reducts=config.max_reducts,
        allow_missing=allow_missing,
        fitting_version=config.fitting_version,
        tree_version=config.tree_version,
        min_sigma_candidates=config.min_sigma_candidates,
        max_sigma_candidates=config.max_sigma_candidates,
        max_sigma_iterations=config.max_sigma_iterations,
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


def _save_postprocessed(name, X, y, allow_missing):
    """Dump the post-preprocessed dataset alongside the raw CSV for inspection.

    Fits the same Preprocessor the classifier uses on the FULL dataset and
    writes datasets/data/{id:02d}_{name}_postprocessing.csv. Per-fold scaling
    differs slightly from this snapshot, which is expected — the file is for
    sanity-checking the preprocessing logic, not for re-feeding the pipeline.
    """
    if name not in NAME_TO_ID:
        return
    pre = Preprocessor(allow_missing=allow_missing)
    X_clean, y_clean = pre.fit_transform(X, y)
    out_path = os.path.join(
        DATA_DIR, f"{NAME_TO_ID[name]:02d}_{name}_postprocessing.csv"
    )
    df = pd.DataFrame(X_clean, columns=[f"f{i}" for i in range(X_clean.shape[1])])
    df["label"] = y_clean
    df.to_csv(out_path, index=False)
    print(f"  Saved postprocessed CSV: {os.path.relpath(out_path)} ({df.shape})")


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

    allow_missing = DATASETS[name].allow_missing_values if name in DATASETS else False
    _save_postprocessed(name, X, y, allow_missing)

    folds = []
    start = time.time()
    for iter_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        fold = _train_one_fold(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx], config, allow_missing
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
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], config, allow_missing
        )
        print(f"  Holdout: CA={holdout_fold.ca:.4f}, MAE={holdout_fold.mae:.4f}")

    result = DatasetResult(
        dataset=name, protocol=protocol, folds=folds,
        elapsed=time.time() - start, n_features=n_features,
        holdout_fold=holdout_fold,
    )
    print_dataset_summary(result)
    return result


def _resolve_log_path(config: BFMDTConfig) -> str:
    """Compose reports/{dataset|all}/v{N}/delta_{delta}/seed_{seed}/{timestamp}.log.

    `v{N}` assumes fitting_version == tree_version (the usual case); falls back
    to `f{fv}t{tv}` if they diverge.
    """
    bucket = config.dataset_names[0] if len(config.dataset_names) == 1 else "all"
    if config.fitting_version == config.tree_version:
        version_label = f"v{config.fitting_version}"
    else:
        version_label = f"f{config.fitting_version}t{config.tree_version}"
    run_dir = os.path.join(
        REPORTS_DIR,
        bucket,
        version_label,
        f"delta_{config.delta}",
        f"seed_{config.random_seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(run_dir, f"{timestamp}.log")


def main():
    config = parse_args()
    log_path = _resolve_log_path(config)

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
