# BFMDT Team Tasks

## Team

| MSHV   | Name     |
|--------|----------|
| 25C15055 | Nguyễn Bùi Mẫn Nhi |
| 25C15045 | Âu Dương Khang |
| 25C15025 | Nguyễn Thiện Thuật |

---

## Tasks

| # | Task | Blocked by | File | Nhi | Khang | Thuật |
|---|------|------------|------|----------|----------|----------|
| 1 | Implement utils (inversion count, ordinal mapping) | -- | `utils.py` | [X] | [ ] | [ ] |
| 2 | Implement preprocessing (imputation, normalization) | -- | `preprocessing.py` | [ ] | [ ] | [ ] |
| 3 | Implement evaluation metrics (CA, MAE) | -- | `metrics.py` | [ ] | [ ] | [ ] |
| 4 | Implement Algorithm 1 (Monotonic Partition) | #1 | `monotonic_partition.py` | [X] | [ ] | [ ] |
| 5 | Implement Algorithm 2 (Fitting Degree Matrix) | #1, #4 | `fitting_degree.py` | [ ] | [ ] | [ ] |
| 6 | Implement sigma candidate selection | #5 | `sigma_selection.py` | [ ] | [ ] | [ ] |
| 7 | Implement Algorithm 3 (Feature Selection) | #5 | `feature_selection.py` | [ ] | [ ] | [ ] |
| 8 | Implement Monotonic Decision Tree (ARMI/DRMI, fit, predict_dsl) | -- | `monotonic_decision_tree.py` | [ ] | [ ] | [ ] |
| 9 | Implement dataset loader + registry | -- | `datasets/` | [ ] | [ ] | [ ] |
| 10 | Implement BFMDTClassifier (fit, predict, predict_proba) | #1-#8 | `bfmdt_classifier.py` | [ ] | [ ] | [ ] |
| 11 | Implement config + CLI argument parsing | #1-#10 | `config.py` | [ ] | [ ] | [ ] |
| 12 | Run benchmarks on datasets 1-9 (small) | #9-#11 | `experiments/run_benchmarks.py` | [ ] | [ ] | [ ] |
| 13 | Run benchmarks on datasets 10-13 (medium) | #9-#11 | `experiments/run_benchmarks.py` | [ ] | [ ] | [ ] |
| 14 | Run benchmarks on datasets 14-18 (large) | #9-#11 | `experiments/run_benchmarks.py` | [ ] | [ ] | [ ] |
| 15 | Compare results with Table 6 & 7 from paper | #12-#14 | | [ ] | [ ] | [ ] |
| 16 | Write final report / presentation | #15 | | [ ] | [ ] | [ ] |

---

## Suggested Plan

**Sprint 1** -- No dependencies, all parallel:

| Member A | Member B | Member C |
|----------|----------|----------|
| #8 Monotonic Decision Tree | #1 Utils | #2 Preprocessing |
| | #4 Monotonic Partition | #3 Metrics |
| | | #9 Dataset loader |

**Sprint 2** -- B & C start after A finishes #5:

| Member A | Member B | Member C |
|----------|----------|----------|
| #5 Fitting Degree Matrix | #6 Sigma Selection | #7 Feature Selection |

**Sprint 3** -- Integrate, config, benchmark in parallel:

| Member A | Member B | Member C |
|----------|----------|----------|
| #10 BFMDTClassifier | (support A) | (support A) |
| #11 Config | | |
| #12 Benchmark small (1-9) | #13 Benchmark medium (10-13) | #14 Benchmark large (14-18) |
| #15 Compare results | #15 Compare results | #15 Compare results |
| #16 Final report | #16 Final report | #16 Final report |
