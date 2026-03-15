# Data Loading Guide

## Overview

This project supports loading various benchmark datasets for machine learning experiments. Datasets are categorized into those available via sklearn.datasets, OpenML, and custom local datasets.

## Available Datasets

### Datasets from sklearn.datasets
- wine

### Datasets from OpenML (with local fallback)
- breast-wisconsin
- breast-cancer
- heart-disease
- hepatitis
- german-credit
- vehicle
- diabetes
- wine-quality
- sonar
- arcene
- wdbc
- divorce
- SMK_CAN_187

### Custom Local Datasets
- turkiye-student (CSV file)
- DrivFace (head-pose dataset)
- Yale (face database)
- PEMS-SF (traffic data)

## Downloading Custom Datasets

For datasets not available in sklearn.datasets or OpenML, you need to download them locally.

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the download script:
   ```
   python datasets/download_datasets.py
   ```
   This will download the datasets from Google Drive to `datasets/data/`.

## Loading Datasets

Use the `load_dataset` function from `datasets.loader`:

```python
from datasets.loader import load_dataset

X, y = load_dataset('wine')  # from sklearn
X, y = load_dataset('breast-wisconsin')  # from OpenML or local
X, y = load_dataset('DrivFace')  # custom local
```

For OpenML datasets, it will try to fetch from OpenML first, then fall back to local CSV if network fails.

For custom datasets, ensure they are downloaded locally.

## Listing Available Datasets

```python
from datasets.loader import list_available_datasets

print(list_available_datasets())
```