"""Custom loader for the UCI breast-cancer dataset (paper ID 3).

LabelEncoder would put '10-14' < '5-9' lexicographically and break monotonicity,
so we ordinal-encode age / tumor-size / inv-nodes by their numeric ranges and
drop the three nominal columns (breast, breast-quad, menopause) that have no
medical ordering. Yields 277 samples × 6 features matching the paper.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_AGE = {
    '20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5,
}
_TUMOR_SIZE = {
    '0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5,
    '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10,
}
_INV_NODES = {
    '0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26': 6,
}
_BINARY = {'no': 0, 'yes': 1}
_CLASS = {'no-recurrence-events': 0, 'recurrence-events': 1}
_DROP_COLS = ['breast', 'breast-quad', 'menopause']


def load_breast_cancer_data(data_dir=DATA_DIR):
    """Load + ordinally encode breast-cancer; returns (X float64, y int64)."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['breast-cancer']:02d}_breast-cancer.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path).dropna().drop(columns=_DROP_COLS)
    df['age'] = df['age'].map(_AGE)
    df['tumor-size'] = df['tumor-size'].map(_TUMOR_SIZE)
    df['inv-nodes'] = df['inv-nodes'].map(_INV_NODES)
    df['node-caps'] = df['node-caps'].map(_BINARY)
    df['irradiat'] = df['irradiat'].map(_BINARY)
    y = df['Class'].map(_CLASS).to_numpy(dtype=np.int64)

    X = df.drop(columns=['Class']).to_numpy(dtype=np.float64)
    return X, y
