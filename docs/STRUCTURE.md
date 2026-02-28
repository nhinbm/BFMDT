# BFMDT Project Structure

## Directory Layout

```
BFMDT/
├── docs/
│   ├── PIPELINE.md                   # Algorithm pipeline (training + prediction)
│   ├── INTERFACES.md                 # Module interfaces (args, returns, types)
│   └── STRUCTURE.md                  # This file
│
├── bfmdt/
│   ├── __init__.py                   # Package exports
│   ├── preprocessing.py              # Step 1: normalization, imputation
│   ├── monotonic_partition.py        # Step 2: Algorithm 1 (AMP/DMP)
│   ├── fitting_degree.py             # Step 3: Algorithm 2 (fitting degree matrix)
│   ├── sigma_selection.py            # Step 4: sigma candidate selection
│   ├── feature_selection.py          # Step 5: Algorithm 3 (monotonic related family)
│   ├── monotonic_decision_tree.py    # Step 6: REMT tree (ARMI/DRMI)
│   ├── bfmdt_classifier.py           # Step 7-8: main classifier (Algorithm 4)
│   ├── metrics.py                    # Evaluation (CA, MAE)
│   └── utils.py                      # Helpers (inversion count, ordinal mapping)
│
├── datasets/
│   ├── __init__.py
│   ├── loader.py                     # Dataset fetcher / cache
│   └── registry.py                   # 18 benchmark datasets metadata
│
├── tests/
│   ├── conftest.py                   # Shared synthetic fixtures
│   ├── test_preprocessing.py
│   ├── test_monotonic_partition.py
│   ├── test_fitting_degree.py
│   ├── test_feature_selection.py
│   ├── test_monotonic_decision_tree.py
│   ├── test_bfmdt_classifier.py
│   └── test_integration.py           # End-to-end on real datasets
│
├── experiments/
│   └── run_benchmarks.py             # Reproduce Table 6/7 from paper
│
├── requirements/
│   ├── base.txt                      # numpy, scipy, scikit-learn, ucimlrepo
│   ├── dev.txt                       # pytest, pytest-cov
│   └── experiment.txt                # pandas, tabulate, matplotlib
│
├── config.py                         # Default hyperparameters and CLI argument parsing
├── pyproject.toml                    # Project metadata, build config
└── .gitignore
```


## Benchmark Datasets

18 datasets from the paper (Table 4).

| ID | Dataset           | Samples | Features | Classes | Source         |
|----|-------------------|---------|----------|---------|----------------|
| 1  | breast-wisconsin  | 683     | 10       | 2       | UCI            |
| 2  | wine              | 178     | 14       | 3       | UCI            |
| 3  | breast-cancer     | 277     | 10       | 2       | UCI            |
| 4  | heart-disease     | 270     | 14       | 2       | UCI            |
| 5  | hepatitis         | 80      | 20       | 2       | UCI            |
| 6  | german-credit     | 1000    | 21       | 2       | UCI            |
| 7  | vehicle           | 946     | 19       | 4       | UCI            |
| 8  | wdbc              | 569     | 31       | 2       | UCI            |
| 9  | diabetes          | 768     | 9        | 2       | UCI            |
| 10 | wine-quality      | 4898    | 12       | 7       | UCI            |
| 11 | divorce           | 170     | 55       | 2       | UCI / AIStudio |
| 12 | sonar             | 208     | 61       | 2       | UCI            |
| 13 | turkiye-student   | 5820    | 33       | 5       | UCI            |
| 14 | Yale              | 165     | 1025     | 15      | jundongl       |
| 15 | arcene            | 200     | 10001    | 2       | UCI            |
| 16 | SMK_CAN_187       | 187     | 19994    | 2       | jundongl       |
| 17 | DrivFace          | 606     | 6401     | 3       | UCI            |
| 18 | PEMS-SF           | 440     | 138673   | 7       | UCI            |

**Evaluation**: 5-fold stratified cross-validation (datasets 1-17), 80/20 split (dataset 18).

**Metrics**: Classification Accuracy (CA), Mean Absolute Error (MAE).