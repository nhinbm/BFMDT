from bfmdt.preprocessing import Preprocessor
from datasets.loader import load_dataset, list_available_datasets
import numpy as np

print("List available datasets:")
available_ds = list_available_datasets()
print(available_ds)

X_raw, y_raw = load_dataset('DrivFace')

print(f" - Feature matrix shape X: {X_raw.shape}")
print(f" - Labels (encoded as integers): {y_raw[:10]}")

preprocessor = Preprocessor()
X_clean, y_clean = preprocessor.fit_transform(X_raw, y_raw)

print(f"[Preprocessor] Dataset after preprocessing:")
print(f" - Feature matrix shape X: {X_clean.shape}")
print(f" - Labels (encoded as integers): {X_clean[:3]}")
print(f" - Total number of missing values (NaN) in X: {np.isnan(X_clean).sum()} -> Filled with Mean!")
print(f" - Min/Max of ALL columns   : Min = {np.min(X_clean)}, Max = {np.max(X_clean)} -> Clamped to range [0, 1]!")
