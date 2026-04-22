"""Reference values from the BFMDT paper for comparison in reporting mode.

All values are taken from Tables 4, 6, 7, 9, 10 of the paper. Keyed by dataset
ID (1-18) to match `config.DATASET_NAMES_BY_ID`.
"""

# Table 4: Number of raw features per dataset
N_FEATURES = {
    1: 10, 2: 14, 3: 10, 4: 14, 5: 20, 6: 21, 7: 19, 8: 31, 9: 9,
    10: 12, 11: 55, 12: 61, 13: 33, 14: 1025, 15: 10001, 16: 19994,
    17: 6401, 18: 138673,
}

# Table 6: BFMDT Classification Accuracy (%) per dataset
CA_PERCENT = {
    1: 95.85, 2: 92.67, 3: 77.24, 4: 71.11, 5: 83.86, 6: 70.00,
    7: 63.70, 8: 94.55, 9: 66.27, 10: 62.52, 11: 99.43, 12: 75.49,
    13: 84.30, 14: 47.42, 15: 56.01, 16: 71.12, 17: 93.57, 18: 99.55,
}

# Table 7: BFMDT Mean Absolute Error (%) per dataset
MAE_PERCENT = {
    1: 4.15, 2: 9.60, 3: 22.76, 4: 28.89, 5: 16.14, 6: 30.00,
    7: 55.20, 8: 5.45, 9: 33.73, 10: 46.00, 11: 0.57, 12: 24.51,
    13: 22.75, 14: 285.40, 15: 43.99, 16: 28.88, 17: 6.43, 18: 0.45,
}

# Table 9: BFMDT average number of feature subsets per dataset
N_REDUCTS = {
    1: 2, 2: 2, 3: 1.5, 4: 1, 5: 1, 6: 2, 7: 1.9, 8: 5, 9: 3, 10: 8,
    11: 1.8, 12: 16, 13: 7, 14: 1, 15: 1.2, 16: 1.6, 17: 13, 18: 2.8,
}

# Table 10: BFMDT total time cost (s), grouped. Each entry is (id_range, label, seconds).
TIME_GROUPS = [
    ((1, 10), "Dataset 1–10 (quy mô nhỏ/vừa)", 5990.93),
    ((11, 11), "Dataset 11 (divorce)", 23.89),
    ((12, 12), "Dataset 12 (sonar)", 133.37),
    ((13, 13), "Dataset 13 (turkiye-student)", 262.89),
    ((14, 16), "Dataset 14–16 (cao chiều)", 3255.38),
    ((17, 17), "Dataset 17 (DrivFace)", 688.18),
    ((18, 18), "Dataset 18 (PEMS-SF)", 4820.32),
]
