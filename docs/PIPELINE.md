# BFMDT Algorithm Pipeline

> Based on: "Fusing Monotonic Decision Tree Based on Related Family"
> IEEE Transactions on Knowledge and Data Engineering, 2024

---

## Training Phase

```
  Raw Data (X, y)
       |
       v
  [Step 1] Preprocessing
       |   - Remove samples with missing decisions
       |   - Mean imputation for missing features
       |   - Range normalization to [0, 1]
       |
       v
  [Step 2] Monotonic Partition  (Algorithm 1)
       |   - For each feature: sort samples, find Maximal Monotonic Intervals (MMIs)
       |   - Produces AMP (ascending) and DMP (descending) per feature
       |
       v
  [Step 3] Fitting Degree Matrix  (Algorithm 2)
       |   - Detect monotone direction per feature (increasing vs decreasing)
       |   - Invert decreasing features
       |   - Compute fitting degree F(x_i, a_j) for every sample-feature pair
       |
       v
  [Step 4] Sigma Candidate Selection
       |   - Find distinct fitting degree values with freq > L/sp
       |   - Adjust sp until 5-30 candidates
       |
       |   For each sigma candidate:
       |   +----------------------------------------------+
       |   |                                              |
       |   v                                              |
       |   [Step 5] Feature Selection  (Algorithm 3)      |
       |   |   - Binarize fitting matrix at sigma         |
       |   |   - Absorb redundant rows                    |
       |   |   - Greedy set cover to find reducts         |
       |   |   -> up to 50 feature subsets                |
       |   |                                              |
       |   |   For each feature subset (reduct):          |
       |   |   +------------------------------------+     |
       |   |   |                                    |     |
       |   |   v                                    |     |
       |   |   [Step 6] Build Trees                 |     |
       |   |   |   - Build ARMI tree (ascending)    |     |
       |   |   |   - Build DRMI tree (descending)   |     |
       |   |   |   -> 2 trees per reduct            |     |
       |   |   +------------------------------------+     |
       |   |                                              |
       |   v                                              |
       |   [Step 7] Fuse & Evaluate                       |
       |   |   - For each test sample: sum DSL from       |
       |   |     all trees, predict argmax class          |
       |   |   - Compute accuracy and MAE                 |
       |   +----------------------------------------------+
       |
       v
  [Step 8] Select Best Sigma
       |   - Keep sigma with highest accuracy
       |   - Store corresponding trees
       |
       v
  Trained BFMDT Model
```

---

## Prediction Phase

```
  New samples X_test
       |
       v
  [Step 1] Normalize
       |   - Apply same normalization from training
       |   - Invert decreasing features
       |
       v
  [Step 2] Traverse All Trees
       |   - For each tree: route sample to leaf node
       |   - Collect DSL (Decision Support Level) per class at each leaf
       |
       v
  [Step 3] Fuse DSL
       |   - Sum DSL across all trees for each class:
       |     Support(Leaf_all, d_k) = SUM Support(Leaf_i, d_k)
       |
       v
  [Step 4] Predict
       |   - d = argmax_k Support(Leaf_all, d_k)
       |
       v
  y_pred
```
