"""Custom loader for the Pima Indians diabetes dataset (paper ID 9).

Converts biological zeros to NaN in 5 columns (plas, pres, skin, insu, mass)
where 0 is medically impossible -- they are missing values disguised as 0 in
the original UCI release. Letting Preprocessor mean-impute downstream avoids
the scaling distortion that arises if 0 is treated as a real measurement.
Yields 768 samples × 8 features. Class: tested_negative=0, tested_positive=1.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_BIOLOGICAL_ZERO_COLS = ['plas', 'pres', 'skin', 'insu', 'mass']
_CLASS = {'tested_negative': 0, 'tested_positive': 1}


def load_diabetes_data(data_dir=DATA_DIR):
    """Load + zero-as-missing diabetes; returns (X float64, y int64). 768 × 8."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['diabetes']:02d}_diabetes.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path)
    df[_BIOLOGICAL_ZERO_COLS] = df[_BIOLOGICAL_ZERO_COLS].replace(0, np.nan)
    y = df['class'].map(_CLASS).to_numpy(dtype=np.int64)

    X = df.drop(columns=['class']).to_numpy(dtype=np.float64)
    return X, y
