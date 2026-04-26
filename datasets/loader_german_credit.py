"""Custom loader for the UCI german-credit (statlog credit-g) dataset (paper ID 6).

Strict approach: drops 5 purely-nominal columns (purpose, personal_status,
other_parties, other_payment_plans, housing) -- these have no defensible
ordinal interpretation and would inject noise into the monotonic pipeline.
Keeps 8 ordinal categoricals (encoded by natural ranking) + 7 numeric features.
Yields 1000 samples × 15 features. Class: good=0 (low risk), bad=1 (high risk).
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_CHECKING_STATUS = {
    '<0': 0, '0<=X<200': 1, '>=200': 2, 'no checking': 3,
}
_CREDIT_HISTORY = {
    'no credits/all paid': 0,
    'all paid': 1,
    'existing paid': 2,
    'delayed previously': 3,
    'critical/other existing credit': 4,
}
_SAVINGS_STATUS = {
    '<100': 0, '100<=X<500': 1, '500<=X<1000': 2, '>=1000': 3,
    'no known savings': 4,
}
_EMPLOYMENT = {
    'unemployed': 0, '<1': 1, '1<=X<4': 2, '4<=X<7': 3, '>=7': 4,
}
_PROPERTY_MAGNITUDE = {
    'no known property': 0, 'car': 1, 'life insurance': 2, 'real estate': 3,
}
_JOB = {
    'unemp/unskilled non res': 0,
    'unskilled resident': 1,
    'skilled': 2,
    'high qualif/self emp/mgmt': 3,
}
_OWN_TELEPHONE = {'none': 0, 'yes': 1}
_FOREIGN_WORKER = {'no': 0, 'yes': 1}
_CLASS = {'good': 0, 'bad': 1}

_DROP_COLS = [
    'purpose', 'personal_status', 'other_parties',
    'other_payment_plans', 'housing',
]


def load_german_credit_data(data_dir=DATA_DIR):
    """Load + ordinally encode german-credit; returns (X float64, y int64)."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['german-credit']:02d}_german-credit.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path).drop(columns=_DROP_COLS)

    df['checking_status'] = df['checking_status'].map(_CHECKING_STATUS)
    df['credit_history'] = df['credit_history'].map(_CREDIT_HISTORY)
    df['savings_status'] = df['savings_status'].map(_SAVINGS_STATUS)
    df['employment'] = df['employment'].map(_EMPLOYMENT)
    df['property_magnitude'] = df['property_magnitude'].map(_PROPERTY_MAGNITUDE)
    df['job'] = df['job'].map(_JOB)
    df['own_telephone'] = df['own_telephone'].map(_OWN_TELEPHONE)
    df['foreign_worker'] = df['foreign_worker'].map(_FOREIGN_WORKER)
    y = df['class'].map(_CLASS).to_numpy(dtype=np.int64)

    X = df.drop(columns=['class']).to_numpy(dtype=np.float64)
    return X, y
