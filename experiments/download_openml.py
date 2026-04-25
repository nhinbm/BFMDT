#!/usr/bin/env python
"""Download breast-wisconsin (OpenML: breast-w) to datasets/data/breast-w.csv.

The loader's local fallback at `datasets.loader.fetch_or_load_local` reads
this CSV when OpenML is unreachable.

Usage:
    python experiments/download_openml.py
    python experiments/download_openml.py --force          # overwrite existing CSV
    python experiments/download_openml.py --data-id 1      # fetch by OpenML data_id
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
    parser.add_argument(
        "--data-id", type=int, default=None,
        help="Fetch by OpenML data_id instead of the default breast-w",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.data_id is not None:
        print(f"[fetch] data_id={args.data_id} from OpenML ...")
        data = fetch_openml(
            data_id=args.data_id,
            as_frame=True,
            parser="auto",
            data_home=OPENML_CACHE_DIR,
        )
        name = data.details.get("name", f"dataset_{args.data_id}")
        csv_path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(csv_path) and not args.force:
            print(f"[skip] {csv_path} already exists (use --force to refetch)")
            return
    else:
        csv_path = CSV_PATH
        if os.path.exists(csv_path) and not args.force:
            print(f"[skip] {csv_path} already exists (use --force to refetch)")
            return
        print(f"[fetch] {OPENML_NAME} from OpenML ...")
        data = fetch_openml(
            name=OPENML_NAME,
            version=1,
            as_frame=True,
            parser="auto",
            data_home=OPENML_CACHE_DIR,
        )

    data.frame.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}  shape={data.frame.shape}")


if __name__ == "__main__":
    main()
