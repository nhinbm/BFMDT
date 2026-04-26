"""Custom loader for the Turkiye Student Evaluation dataset (paper ID 13).

Derives the target from mean(Q1-Q28) per Gunduz & Fokoue (2015), the original
paper that introduced this dataset. Each student's overall satisfaction is the
mean of their 28 Likert ratings, binned into 3 classes:

    [1.0, 2.5)  -> 0  Dissatisfied
    [2.5, 3.5)  -> 1  Neutral
    [3.5, 5.0]  -> 2  Satisfied

Target is derived (not extracted from an existing column), so all 33 original
columns (instr, class, nb.repeat, attendance, difficulty, Q1-Q28) are kept as
features -- matches paper Table 4's reported 33 features. Note: target is
computed from Q1-Q28, so the model can in principle recover the binning rule
from those columns -- consistent with Gunduz & Fokoue's original setup.
Yields 5820 samples × 33 features.
"""

import os

import numpy as np
import pandas as pd

from config import DATA_DIR, NAME_TO_ID


_Q_COLS = [f'Q{i}' for i in range(1, 29)]
_BINS = [1.0, 2.5, 3.5, 5.0 + 1e-9]


def load_turkiye_student_data(data_dir=DATA_DIR):
    """Load + derive 3-class satisfaction target; returns (X float64, y int64)."""
    file_path = os.path.join(
        data_dir, f"{NAME_TO_ID['turkiye-student']:02d}_turkiye-student.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    df = pd.read_csv(file_path)
    q_mean = df[_Q_COLS].mean(axis=1)
    y = pd.cut(q_mean, bins=_BINS, labels=[0, 1, 2], right=False).astype(np.int64).to_numpy()

    X = df.to_numpy(dtype=np.float64)
    return X, y
