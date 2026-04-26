"""Custom loader for the UCI heart-disease (statlog) dataset (paper ID 4).

Re-encodes `thal` {3, 6, 7} -> {0, 1, 2} so the three thalassemia categories sit
at equal min-max distances rather than at 0.00 / 0.75 / 1.00 (3/6/7 are medical
codes, not measurements). All other features are already numeric and left for
the Preprocessor to min-max scale per fold. 270 samples × 13 features.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_THAL = {3: 0, 6: 1, 7: 2}
_CLASS = {'absent': 0, 'present': 1}


def load_heart_disease_data(data_dir=DATA_DIR):
    """Load + re-encode heart-disease; returns (X float64, y int64)."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['heart-disease']:02d}_heart-disease.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path)
    df['thal'] = df['thal'].map(_THAL)
    y = df['class'].map(_CLASS).to_numpy(dtype=np.int64)

    X = df.drop(columns=['class']).to_numpy(dtype=np.float64)
    return X, y
