"""Custom loader for the Statlog vehicle silhouettes dataset (paper ID 7).

Encodes the 4 vehicle classes by silhouette size (smallest -> largest):
opel saloon (0), saab saloon (1), van (2), double-decker bus (3). This gives
the decision an ordinal monotonic structure that BFMDT can exploit; alphabetical
LabelEncoder would put bus first and break monotonicity. All 18 geometric
features are already integer and pass through to the Preprocessor.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_CLASS = {'opel': 0, 'saab': 1, 'van': 2, 'bus': 3}


def load_vehicle_data(data_dir=DATA_DIR):
    """Load + ordinally encode vehicle; returns (X float64, y int64). 846 × 18."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['vehicle']:02d}_vehicle.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path)
    y = df['Class'].map(_CLASS).to_numpy(dtype=np.int64)
    X = df.drop(columns=['Class']).to_numpy(dtype=np.float64)
    return X, y
