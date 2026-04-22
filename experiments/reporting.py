"""Result types and output helpers for benchmark runs.

Holds the `FoldResult` / `DatasetResult` dataclasses produced by a benchmark
run, together with the console printers that consume them. Kept separate
from run_benchmarks.py so benchmark orchestration does not drift into
formatting concerns.
"""

from dataclasses import dataclass

import numpy as np

from config import NAME_TO_ID
import paper_reference as paper


@dataclass
class FoldResult:
    """Metrics from fitting and evaluating one train/test split."""
    ca: float
    mae: float
    best_sigma: float | None
    n_sigma_candidates: int | None
    n_reducts: int | None


@dataclass
class DatasetResult:
    """Aggregated results for one dataset across all folds/splits.

    `folds` holds the main-protocol results (CV for most datasets, holdout for
    PEMS-SF). `holdout_fold` is a separate 80:20 holdout result used specifically
    for σ analysis (paper §5.4.1 convention). Populated only in reporting mode
    when the main protocol is CV and σ table is requested.
    """
    dataset: str
    protocol: str
    folds: list[FoldResult]
    elapsed: float
    n_features: int = 0
    holdout_fold: FoldResult | None = None


def print_dataset_summary(result: DatasetResult):
    """Print aggregated metrics and timing for one dataset."""
    cas = np.array([f.ca for f in result.folds])
    maes = np.array([f.mae for f in result.folds])

    if len(result.folds) == 1:
        fold = result.folds[0]
        print(f"\n  CA:  {fold.ca:.4f}")
        print(f"  MAE: {fold.mae:.4f}")
        if fold.best_sigma is not None:
            print(
                f"  σ_best: {fold.best_sigma:.4f}  "
                f"(from {fold.n_sigma_candidates} candidates)"
            )
    else:
        print(f"\n  Mean CA:  {cas.mean():.4f} +/- {cas.std():.4f}")
        print(f"  Mean MAE: {maes.mean():.4f} +/- {maes.std():.4f}")
    print(f"  Time: {result.elapsed:.1f}s")


def print_sigma_analysis_table(results: list[DatasetResult]):
    """Print Bảng 4.5.1.1: σ_best summary across datasets.

    Uses `holdout_fold` if available (paper §5.4.1: 80:20 holdout). Falls
    back to single-fold `folds[0]` (for PEMS-SF) or CV mean±std otherwise.
    """
    print(f"\n{'=' * 95}")
    print("Bảng 4.5.1.1: Kết quả σ_best tối ưu trên từng bộ dữ liệu (80:20 holdout)")
    print(f"{'=' * 95}")
    header = (
        f"{'ID':<4} {'Dataset':<20} {'σ_best':<18} {'N':<10} "
        f"{'CA (%)':<14} {'MAE (%)':<14}"
    )
    print(header)
    print("-" * 95)

    for r in results:
        ds_id = NAME_TO_ID.get(r.dataset, "?")
        sigma_fold = r.holdout_fold or (r.folds[0] if len(r.folds) == 1 else None)

        if sigma_fold is not None:
            sigma_str = (
                f"{sigma_fold.best_sigma:.4f}"
                if sigma_fold.best_sigma is not None else "N/A"
            )
            n_str = (
                str(sigma_fold.n_sigma_candidates)
                if sigma_fold.n_sigma_candidates else "N/A"
            )
            ca_str = f"{sigma_fold.ca * 100:.2f}"
            mae_str = f"{sigma_fold.mae * 100:.2f}"
        else:
            sigmas = np.array([f.best_sigma for f in r.folds if f.best_sigma is not None])
            ncands = np.array(
                [f.n_sigma_candidates for f in r.folds if f.n_sigma_candidates]
            )
            cas = np.array([f.ca for f in r.folds]) * 100
            maes = np.array([f.mae for f in r.folds]) * 100
            sigma_str = (
                f"{sigmas.mean():.4f}±{sigmas.std():.3f}" if len(sigmas) else "N/A"
            )
            n_str = f"{ncands.mean():.1f}" if len(ncands) else "N/A"
            ca_str = f"{cas.mean():.2f}±{cas.std():.2f}"
            mae_str = f"{maes.mean():.2f}±{maes.std():.2f}"

        print(
            f"{ds_id:<4} {r.dataset:<20} {sigma_str:<18} {n_str:<10} "
            f"{ca_str:<14} {mae_str:<14}"
        )
    print("=" * 95)


def _status_label(delta_ca, delta_mae):
    """Classify by how much WORSE than paper; matching or beating paper is OK.

    Only penalizes negative ΔCA (CA below paper) and positive ΔMAE (MAE above paper).
    Exceeding paper in either direction is always OK.

    - worse_by ≤ 2%  → OK
    - worse_by 2-5%  → Chấp nhận được
    - worse_by > 5%  → Cần kiểm tra
    """
    worse_by = max(max(0.0, -delta_ca), max(0.0, delta_mae))
    if worse_by <= 2.0:
        return "OK"
    if worse_by <= 5.0:
        return "Chấp nhận được"
    return "Cần kiểm tra"


def print_ca_mae_table(results: list[DatasetResult]):
    """Print Table 6+7 combined: CA and MAE comparison vs paper."""
    print(f"\n{'=' * 110}")
    print("Bảng 6+7: So sánh CA và MAE với paper")
    print(f"{'=' * 110}")
    header = (
        f"{'ID':<4} {'Dataset':<20} "
        f"{'CA paper':>10} {'CA bc':>12} {'Δ CA':>10} "
        f"{'MAE paper':>10} {'MAE bc':>12} {'Δ MAE':>10} "
        f"{'Trạng thái':<16}"
    )
    print(header)
    print("-" * 110)

    ca_deltas = []
    mae_deltas = []
    ca_reports = []
    mae_reports = []

    for r in results:
        ds_id = NAME_TO_ID.get(r.dataset, None)
        if ds_id is None:
            continue
        cas = np.array([f.ca for f in r.folds]) * 100
        maes = np.array([f.mae for f in r.folds]) * 100
        ca_mean, ca_std = cas.mean(), cas.std()
        mae_mean, mae_std = maes.mean(), maes.std()

        ca_paper = paper.CA_PERCENT[ds_id]
        mae_paper = paper.MAE_PERCENT[ds_id]
        delta_ca = ca_mean - ca_paper
        delta_mae = mae_mean - mae_paper

        ca_reports.append(ca_mean)
        mae_reports.append(mae_mean)
        ca_deltas.append(delta_ca)
        mae_deltas.append(delta_mae)

        ca_bc = f"{ca_mean:.2f} ±{ca_std:.2f}"
        mae_bc = f"{mae_mean:.2f} ±{mae_std:.2f}"
        print(
            f"{ds_id:<4} {r.dataset:<20} "
            f"{ca_paper:>10.2f} {ca_bc:>12} {delta_ca:>+10.2f} "
            f"{mae_paper:>10.2f} {mae_bc:>12} {delta_mae:>+10.2f} "
            f"{_status_label(delta_ca, delta_mae):<16}"
        )

    if results:
        print("-" * 110)
        ca_paper_avg = np.mean([paper.CA_PERCENT[NAME_TO_ID[r.dataset]] for r in results])
        mae_paper_avg = np.mean([paper.MAE_PERCENT[NAME_TO_ID[r.dataset]] for r in results])
        print(
            f"{'':4} {'Trung bình':<20} "
            f"{ca_paper_avg:>10.2f} {np.mean(ca_reports):>12.2f} {np.mean(ca_deltas):>+10.2f} "
            f"{mae_paper_avg:>10.2f} {np.mean(mae_reports):>12.2f} {np.mean(mae_deltas):>+10.2f}"
        )
    print("=" * 110)


def print_reducts_table(results: list[DatasetResult]):
    """Print Table 9: number of feature subsets vs paper."""
    print(f"\n{'=' * 95}")
    print("Bảng 9: So sánh số lượng feature subsets với paper")
    print(f"{'=' * 95}")
    header = (
        f"{'ID':<4} {'Dataset':<20} "
        f"{'Số feature gốc':>15} {'Reducts bc':>12} {'Reducts paper':>14} {'Chênh lệch':>12}"
    )
    print(header)
    print("-" * 95)

    for r in results:
        ds_id = NAME_TO_ID.get(r.dataset, None)
        if ds_id is None:
            continue
        reducts = np.array([f.n_reducts for f in r.folds if f.n_reducts is not None])
        if len(reducts) == 0:
            continue
        reducts_mean = reducts.mean()
        reducts_paper = paper.N_REDUCTS[ds_id]
        delta = reducts_mean - reducts_paper
        print(
            f"{ds_id:<4} {r.dataset:<20} "
            f"{r.n_features:>15} {reducts_mean:>12.2f} "
            f"{reducts_paper:>14} {delta:>+12.2f}"
        )
    print("=" * 95)


def print_time_table(results: list[DatasetResult]):
    """Print Table 10: total time grouped by dataset ranges vs paper."""
    print(f"\n{'=' * 85}")
    print("Bảng 10: So sánh thời gian thực thi với paper")
    print(f"{'=' * 85}")
    header = (
        f"{'Nhóm dataset':<38} "
        f"{'Thời gian bc (s)':>18} {'Paper (s)':>12} {'Chênh lệch (%)':>14}"
    )
    print(header)
    print("-" * 85)

    time_by_id = {
        NAME_TO_ID[r.dataset]: r.elapsed
        for r in results if r.dataset in NAME_TO_ID
    }

    for (start_id, end_id), label, paper_seconds in paper.TIME_GROUPS:
        group_time = sum(time_by_id.get(i, 0.0) for i in range(start_id, end_id + 1))
        if group_time == 0.0:
            continue
        pct = (group_time - paper_seconds) / paper_seconds * 100
        print(
            f"{label:<38} "
            f"{group_time:>18.2f} {paper_seconds:>12.2f} {pct:>+14.2f}"
        )
    print("=" * 85)
