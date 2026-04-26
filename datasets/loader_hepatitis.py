"""Custom loader for the UCI hepatitis dataset (paper ID 5).

Drops rows with any missing value -- 75 of 155 rows have at least one NaN
(PROTIME alone is 43% missing), so mean-imputation would dominate the signal.
Yields 80 samples × 19 features. All binary categoricals (SEX + 12 yes/no
symptom columns) are mapped to 0/1; numeric columns (AGE + 5 continuous lab
values) pass through for the Preprocessor to min-max scale per fold.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_SEX = {'female': 0, 'male': 1}
_YES_NO = {'no': 0, 'yes': 1}
_CLASS = {'LIVE': 0, 'DIE': 1}
_YES_NO_COLS = [
    'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA',
    'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE', 'SPIDERS',
    'ASCITES', 'VARICES', 'HISTOLOGY',
]


def load_hepatitis_data(data_dir=DATA_DIR):
    """Load + encode hepatitis; returns (X float64, y int64). 80 × 19 after dropna."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['hepatitis']:02d}_hepatitis.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path).dropna()
    df['SEX'] = df['SEX'].map(_SEX)
    for col in _YES_NO_COLS:
        df[col] = df[col].map(_YES_NO)
    y = df['Class'].map(_CLASS).to_numpy(dtype=np.int64)

    X = df.drop(columns=['Class']).to_numpy(dtype=np.float64)
    return X, y
