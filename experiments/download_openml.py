#!/usr/bin/env python
"""Download an OpenML dataset (by paper Table 4 ID) to datasets/data/.

The loader's local fallback at `datasets.loader.fetch_or_load_local` reads
this CSV when OpenML is unreachable.

Usage:
    python experiments/download_openml.py                  # default: id=1 (breast-wisconsin)
    python experiments/download_openml.py --data-id 2      # fetch id=2 (wine)
    python experiments/download_openml.py --data-id 1 --force
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import fetch_openml

from config import DATA_DIR, DATASET_NAMES_BY_ID, OPENML_CACHE_DIR
from datasets.loader import OPENML_DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Download an OpenML dataset (by paper Table 4 ID) to datasets/data/"
    )
    parser.add_argument(
        "--data-id", type=int, default=1,
        help="Paper Table 4 dataset ID (default: 1 = breast-wisconsin)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the CSV already exists",
    )
    args = parser.parse_args()

    if args.data_id not in DATASET_NAMES_BY_ID:
        parser.error(
            f"--data-id must be one of {sorted(DATASET_NAMES_BY_ID)}"
        )
    project_name = DATASET_NAMES_BY_ID[args.data_id]
    if project_name not in OPENML_DATASETS:
        parser.error(
            f"id={args.data_id} ('{project_name}') is not on OpenML; "
            f"use the dedicated downloader for that dataset."
        )
    openml_name = OPENML_DATASETS[project_name]
    csv_path = os.path.join(DATA_DIR, f"{args.data_id:02d}_{project_name}.csv")

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(csv_path) and not args.force:
        print(f"[skip] {csv_path} already exists (use --force to refetch)")
        return

    print(f"[fetch] id={args.data_id} '{project_name}' (OpenML: {openml_name}) ...")
    data = fetch_openml(
        name=openml_name,
        version=1,
        as_frame=True,
        parser="auto",
        data_home=OPENML_CACHE_DIR,
    )
    data.frame.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}  shape={data.frame.shape}")


if __name__ == "__main__":
    main()
