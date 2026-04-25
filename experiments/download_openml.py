#!/usr/bin/env python
"""Download breast-wisconsin (OpenML: breast-w) to datasets/data/breast-w.csv.

The loader's local fallback at `datasets.loader.fetch_or_load_local` reads
this CSV when OpenML is unreachable.

Usage:
    python experiments/download_openml.py
    python experiments/download_openml.py --force   # overwrite existing CSV
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import fetch_openml

from config import DATA_DIR, OPENML_CACHE_DIR

OPENML_NAME = "breast-w"
CSV_PATH = os.path.join(DATA_DIR, f"{OPENML_NAME}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Download breast-wisconsin from OpenML to datasets/data/"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the CSV already exists",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and not args.force:
        print(f"[skip] {CSV_PATH} already exists (use --force to refetch)")
        return

    print(f"[fetch] {OPENML_NAME} from OpenML ...")
    data = fetch_openml(
        name=OPENML_NAME,
        version=1,
        as_frame=True,
        parser="auto",
        data_home=OPENML_CACHE_DIR,
    )
    data.frame.to_csv(CSV_PATH, index=False)
    print(f"[save]  {CSV_PATH}  shape={data.frame.shape}")


if __name__ == "__main__":
    main()
