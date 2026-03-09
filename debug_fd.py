"""Debug script to trace fitting degree computation and find discrepancies."""

import numpy as np
import sys
sys.path.insert(0, '.')

from bfmdt.monotonic_partition import MonotonicPartitioner
from bfmdt.fitting_degree import FittingDegreeComputer
from bfmdt.utils import compute_inversion_count_for_feature, count_inversions

# --- Test data from the paper (Table 1) ---
X = np.array([
    [0.1, 0.0, 0.3, 0.2, 1.0, 0.4, 0.6, 0.8],
    [0.3, 0.1, 0.6, 0.0, 0.6, 0.2, 0.1, 0.9],
    [0.3, 0.0, 0.5, 0.5, 0.9, 0.0, 0.7, 0.6],
    [0.1, 0.3, 0.3, 0.5, 0.5, 0.1, 0.8, 0.7],
    [0.3, 0.2, 0.7, 0.7, 0.0, 0.4, 0.4, 0.5],
    [0.0, 0.5, 0.1, 1.0, 0.1, 0.5, 0.1, 0.3],
    [0.6, 0.5, 0.1, 0.8, 0.6, 0.5, 0.0, 0.2],
    [0.1, 0.6, 1.0, 1.0, 0.2, 0.7, 0.3, 0.2],
    [0.1, 1.0, 1.0, 0.7, 0.1, 1.0, 1.0, 0.0],
])
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

expected = np.array([
    [0.23, 1.0, 0.24, 0.94, 0.35, 0.63, 0.41, 0.19],
    [0.46, 1.0, 0.48, 0.94, 0.18, 0.63, 0.2,  0.19],
    [0.46, 1.0, 0.48, 0.94, 0.35, 0.31, 0.41, 0.09],
    [0.23, 1.0, 0.24, 0.63, 0.53, 0.31, 0.2,  0.09],
    [0.23, 1.0, 0.24, 0.63, 0.53, 0.63, 0.2,  0.19],
    [0.23, 1.0, 0.24, 0.31, 0.53, 0.63, 0.2,  0.19],
    [0.23, 1.0, 0.24, 0.63, 0.18, 0.94, 0.2,  0.28],
    [0.46, 1.0, 0.48, 0.31, 0.35, 0.94, 0.2,  0.28],
    [0.46, 1.0, 0.48, 0.63, 0.35, 0.94, 0.2,  0.28],
])

n_samples, n_features = X.shape
classes, class_counts = np.unique(y, return_counts=True)
class_size = dict(zip(classes, class_counts))
denom = n_samples * (n_samples - 1)  # 72

print("=" * 80)
print("SETUP")
print("=" * 80)
print(f"n_samples = {n_samples}, n_features = {n_features}")
print(f"classes = {classes}, class_counts = {class_counts}")
print(f"class_size = {class_size}")
print(f"denom = N*(N-1) = {denom}")
print()

# --- Step 1: Run MonotonicPartitioner ---
mp = MonotonicPartitioner().fit(X, y)

print("=" * 80)
print("MONOTONIC PARTITIONS")
print("=" * 80)
for j in range(n_features):
    print(f"\n--- Feature {j} ---")
    print(f"  AMP (ascending):")
    for k, mmi in enumerate(mp.ascending_partitions[j]):
        print(f"    MMI[{k}]: indices={mmi}, X_vals={X[mmi, j]}, y_vals={y[mmi]}")
    print(f"  DMP (descending):")
    for k, mmi in enumerate(mp.descending_partitions[j]):
        print(f"    MMI[{k}]: indices={mmi}, X_vals={X[mmi, j]}, y_vals={y[mmi]}")

# --- Step 2: Run FittingDegreeComputer ---
fd = FittingDegreeComputer().fit(X, y, mp.ascending_partitions, mp.descending_partitions)

print("\n" + "=" * 80)
print("FITTING DEGREE MATRIX (actual)")
print("=" * 80)
print(np.round(fd.matrix, 4))
print(f"\nMonotone directions: {fd.monotone_directions}")

print("\n" + "=" * 80)
print("EXPECTED FITTING DEGREE MATRIX")
print("=" * 80)
print(expected)

print("\n" + "=" * 80)
print("DIFFERENCE (actual - expected)")
print("=" * 80)
print(np.round(fd.matrix - expected, 4))

# --- Step 3: Detailed trace for feature 0 (j=0) ---
def trace_feature(j, label):
    print("\n" + "=" * 80)
    print(f"DETAILED TRACE: Feature {j} ({label})")
    print("=" * 80)

    # Feature values and sort order
    feature_vals = X[:, j]
    print(f"\nFeature values: {feature_vals}")
    print(f"y values:       {y}")

    # Ascending inversion count
    sorted_asc = np.lexsort((y, X[:, j]))
    y_sorted_asc = y[sorted_asc]
    print(f"\nASCENDING direction:")
    print(f"  Sorted indices (lexsort by (y, X[:,{j}])): {sorted_asc}")
    print(f"  X values in sorted order: {X[sorted_asc, j]}")
    print(f"  y values in sorted order: {y_sorted_asc}")

    inv_asc = compute_inversion_count_for_feature(X, y, j, ascending=True)
    global_factor_asc = 1.0 - 2.0 * inv_asc / denom
    print(f"  Inversion count (ascending): {inv_asc}")
    print(f"  Global factor (ascending): 1 - 2*{inv_asc}/{denom} = {global_factor_asc:.6f}")

    # Let's also manually count inversions for ascending
    y_sorted_copy = y_sorted_asc.astype(float).copy()
    manual_inv = 0
    for i in range(len(y_sorted_copy)):
        for jj in range(i+1, len(y_sorted_copy)):
            if y_sorted_copy[i] > y_sorted_copy[jj]:
                manual_inv += 1
    print(f"  Manual inversion count (brute force): {manual_inv}")

    # Descending inversion count
    sorted_desc = np.lexsort((-y, X[:, j]))
    # For descending, the implementation negates y
    sorted_desc_impl = np.lexsort((-y, X[:, j]))
    y_sorted_desc_impl = (-y[sorted_desc_impl]).astype(float).copy()
    print(f"\nDESCENDING direction:")
    print(f"  Sorted indices (lexsort by (-y, X[:,{j}])): {sorted_desc_impl}")
    print(f"  X values in sorted order: {X[sorted_desc_impl, j]}")
    print(f"  y values in sorted order: {y[sorted_desc_impl]}")
    print(f"  Negated y in sorted order (used for inv count): {y_sorted_desc_impl}")

    inv_desc = compute_inversion_count_for_feature(X, y, j, ascending=False)
    global_factor_desc = 1.0 - 2.0 * inv_desc / denom
    print(f"  Inversion count (descending): {inv_desc}")
    print(f"  Global factor (descending): 1 - 2*{inv_desc}/{denom} = {global_factor_desc:.6f}")

    # Ascending local parts
    print(f"\nASCENDING LOCAL PARTS:")
    partition_asc = mp.ascending_partitions[j]
    F_asc = np.zeros(n_samples)
    for k, mmi in enumerate(partition_asc):
        mmi_y = y[mmi]
        unique_classes, inverse, counts = np.unique(mmi_y, return_inverse=True, return_counts=True)
        same_class_counts = counts[inverse]
        class_sizes_for_mmi = np.array([class_size[c] for c in unique_classes])
        local_parts = same_class_counts / class_sizes_for_mmi[inverse]
        print(f"  MMI[{k}]: indices={mmi}")
        print(f"    y_vals={mmi_y}")
        print(f"    unique_classes={unique_classes}, counts_in_mmi={counts}")
        print(f"    For each sample in MMI:")
        for idx_in_mmi, sample_idx in enumerate(mmi):
            sc = same_class_counts[idx_in_mmi]
            cs = class_sizes_for_mmi[inverse[idx_in_mmi]]
            lp = local_parts[idx_in_mmi]
            print(f"      sample {sample_idx}: class={y[sample_idx]}, "
                  f"same_class_in_mmi={sc}, class_total={cs}, "
                  f"local_part={lp:.4f}, "
                  f"F = {lp:.4f} * {global_factor_asc:.4f} = {lp * global_factor_asc:.4f}")
        F_asc[mmi] = local_parts * global_factor_asc

    # Descending local parts
    print(f"\nDESCENDING LOCAL PARTS:")
    partition_desc = mp.descending_partitions[j]
    F_desc = np.zeros(n_samples)
    for k, mmi in enumerate(partition_desc):
        mmi_y = y[mmi]
        unique_classes, inverse, counts = np.unique(mmi_y, return_inverse=True, return_counts=True)
        same_class_counts = counts[inverse]
        class_sizes_for_mmi = np.array([class_size[c] for c in unique_classes])
        local_parts = same_class_counts / class_sizes_for_mmi[inverse]
        print(f"  MMI[{k}]: indices={mmi}")
        print(f"    y_vals={mmi_y}")
        print(f"    unique_classes={unique_classes}, counts_in_mmi={counts}")
        print(f"    For each sample in MMI:")
        for idx_in_mmi, sample_idx in enumerate(mmi):
            sc = same_class_counts[idx_in_mmi]
            cs = class_sizes_for_mmi[inverse[idx_in_mmi]]
            lp = local_parts[idx_in_mmi]
            print(f"      sample {sample_idx}: class={y[sample_idx]}, "
                  f"same_class_in_mmi={sc}, class_total={cs}, "
                  f"local_part={lp:.4f}, "
                  f"F = {lp:.4f} * {global_factor_desc:.4f} = {lp * global_factor_desc:.4f}")
        F_desc[mmi] = local_parts * global_factor_desc

    print(f"\nF_asc for feature {j}: {np.round(F_asc, 4)}")
    print(f"F_desc for feature {j}: {np.round(F_desc, 4)}")
    print(f"sum(F_asc) = {np.sum(F_asc):.4f}, sum(F_desc) = {np.sum(F_desc):.4f}")
    chosen = "ascending" if np.sum(F_asc) >= np.sum(F_desc) else "descending"
    print(f"Chosen direction: {chosen}")
    print(f"Expected values: {expected[:, j]}")
    print(f"Actual values:   {np.round(fd.matrix[:, j], 4)}")

trace_feature(0, "a1")
trace_feature(4, "a5")

# --- Also trace feature 1 (j=1) since it's perfectly 1.0 ---
trace_feature(1, "a2")

# --- Step 4: Check the paper's formula interpretation ---
print("\n" + "=" * 80)
print("ANALYSIS: Paper formula vs implementation")
print("=" * 80)
print("""
Paper formula (Eq. 19-22):
  F(x_i, a_j) = local_part * global_factor

  local_part = |[x_i]_D ∩ MMI_k| / |[x_i]_D|
    where [x_i]_D = set of all samples with same class as x_i
    and MMI_k = the MMI containing x_i

  global_factor = 1 - 2*inv / (N*(N-1))
    where inv = inversion count when sorting by feature a_j

Implementation:
  local_part = same_class_counts / class_sizes_for_mmi[inverse]
    same_class_counts = count of x_i's class within the MMI
    class_sizes_for_mmi[inverse] = total count of x_i's class in full dataset

  This matches: |[x_i]_D ∩ MMI_k| / |[x_i]_D|  -- CORRECT interpretation
""")

# --- Step 5: Check what happens with the partition structure more carefully ---
print("=" * 80)
print("PARTITION VERIFICATION")
print("=" * 80)
print("\nFor ascending partition, the implementation does:")
print("  sorted_indices = np.lexsort((y, X[:, j]))")
print("  Then splits where y_sorted[i] > y_sorted[i+1]")
print()
print("This means samples are sorted by feature value first, then by y as tiebreaker.")
print("MMIs are formed where y is non-decreasing along this sorted order.")
print()
print("QUESTION: Does the paper sort by feature value only (ignoring y for ties)?")
print("The lexsort tiebreaker using y might change which samples end up in which MMI.")

# --- Step 6: Alternative computation without lexsort tiebreaker ---
print("\n" + "=" * 80)
print("ALTERNATIVE: Sort by feature value ONLY (no y tiebreaker)")
print("=" * 80)

def partition_feature_no_tiebreak(X, y, j, direction):
    """Sort by feature value only, then check monotonicity of y."""
    n = X.shape[0]
    if direction == "ascending":
        sorted_indices = np.argsort(X[:, j], kind='stable')
    else:
        sorted_indices = np.argsort(-X[:, j], kind='stable')

    y_sorted = y[sorted_indices]

    if direction == "ascending":
        breaks = y_sorted[:-1] > y_sorted[1:]
    else:
        breaks = y_sorted[:-1] < y_sorted[1:]

    split_points = np.where(breaks)[0] + 1
    mmis = np.split(sorted_indices, split_points)
    return list(mmis)

for j in [0, 4]:
    print(f"\n--- Feature {j} ---")
    alt_asc = partition_feature_no_tiebreak(X, y, j, "ascending")
    impl_asc = mp.ascending_partitions[j]
    print(f"  Implementation AMP:")
    for k, mmi in enumerate(impl_asc):
        print(f"    MMI[{k}]: {mmi} -> y={y[mmi]}")
    print(f"  Alternative AMP (no y tiebreak):")
    for k, mmi in enumerate(alt_asc):
        print(f"    MMI[{k}]: {mmi} -> y={y[mmi]}")

# --- Step 7: Check inversion count computation more carefully ---
print("\n" + "=" * 80)
print("INVERSION COUNT ANALYSIS")
print("=" * 80)

for j in range(n_features):
    inv_asc = compute_inversion_count_for_feature(X, y, j, ascending=True)
    inv_desc = compute_inversion_count_for_feature(X, y, j, ascending=False)
    gf_asc = 1.0 - 2.0 * inv_asc / denom
    gf_desc = 1.0 - 2.0 * inv_desc / denom
    print(f"  Feature {j}: inv_asc={inv_asc:2d}, gf_asc={gf_asc:.4f} | "
          f"inv_desc={inv_desc:2d}, gf_desc={gf_desc:.4f}")

# From expected values, let's reverse-engineer what the global factors should be
print("\n" + "=" * 80)
print("REVERSE-ENGINEERING EXPECTED GLOBAL FACTORS")
print("=" * 80)
print("If local_part for sample 0, feature 0 is 1/3 (one class-1 sample in MMI with 3 class-1 total):")
print(f"  expected F = 0.23, so global_factor = 0.23 / (1/3) = {0.23 / (1/3):.4f}")
print("If local_part for sample 1, feature 0 is 2/3:")
print(f"  expected F = 0.46, so global_factor = 0.46 / (2/3) = {0.46 / (2/3):.4f}")
print()
print("So global factor for feature 0 should be ~0.69")
print(f"That means 1 - 2*inv/72 = 0.69, so inv = (1-0.69)*72/2 = {(1-0.69)*72/2:.1f}")

# --- Step 8: What if the inversion count ignores ties in feature values? ---
print("\n" + "=" * 80)
print("INVERSION COUNT: Should pairs with EQUAL feature values be excluded?")
print("=" * 80)

for j in [0, 4]:
    feature_vals = X[:, j]
    print(f"\nFeature {j}: values = {feature_vals}")

    # Count all pairs, pairs with equal features, pairs with strictly ordered features
    total_pairs = 0
    concordant = 0
    discordant = 0
    tied_feature = 0

    for i in range(n_samples):
        for jj in range(i+1, n_samples):
            total_pairs += 1
            if feature_vals[i] == feature_vals[jj]:
                tied_feature += 1
            elif (feature_vals[i] < feature_vals[jj] and y[i] > y[jj]) or \
                 (feature_vals[i] > feature_vals[jj] and y[i] < y[jj]):
                discordant += 1
            elif (feature_vals[i] < feature_vals[jj] and y[i] <= y[jj]) or \
                 (feature_vals[i] > feature_vals[jj] and y[i] >= y[jj]):
                concordant += 1

    print(f"  Total pairs: {total_pairs}")
    print(f"  Tied feature pairs: {tied_feature}")
    print(f"  Concordant (asc): {concordant}")
    print(f"  Discordant (asc): {discordant}")
    print(f"  If global_factor = 1 - 2*discordant/N(N-1): {1.0 - 2.0*discordant/denom:.4f}")
    print(f"  If global_factor = 1 - 2*discordant/(N(N-1) - tied): {1.0 - 2.0*discordant/(denom - 2*tied_feature):.4f}" if denom - 2*tied_feature > 0 else "")

    # What the implementation actually computes
    inv_actual = compute_inversion_count_for_feature(X, y, j, ascending=True)
    print(f"  Implementation inversion count: {inv_actual}")
    print(f"  Implementation global_factor: {1.0 - 2.0*inv_actual/denom:.4f}")

# --- Step 9: What about the descending partition for feature 4? ---
print("\n" + "=" * 80)
print("FEATURE 4 DEEP DIVE")
print("=" * 80)
j = 4
feature_vals = X[:, j]
print(f"Feature 4 values: {feature_vals}")
print(f"y values:         {y}")
print(f"Expected FD:      {expected[:, j]}")
print(f"Actual FD:        {np.round(fd.matrix[:, j], 4)}")
print()

# Manually sort by feature 4 ascending
order_asc = np.argsort(X[:, j])
print(f"Samples sorted by feature 4 (ascending):")
for idx in order_asc:
    print(f"  sample {idx}: X={X[idx, j]}, y={y[idx]}")

# Now check what direction was chosen
print(f"\nChosen direction for feature 4: {fd.monotone_directions[j]}")

# Check both directions
print(f"\nAscending partitions for feature 4:")
for k, mmi in enumerate(mp.ascending_partitions[j]):
    print(f"  MMI[{k}]: indices={mmi}, X={X[mmi, j]}, y={y[mmi]}")
print(f"\nDescending partitions for feature 4:")
for k, mmi in enumerate(mp.descending_partitions[j]):
    print(f"  MMI[{k}]: indices={mmi}, X={X[mmi, j]}, y={y[mmi]}")

# --- Step 10: Recompute with expected global factor ---
print("\n" + "=" * 80)
print("RECOMPUTATION WITH REVERSE-ENGINEERED GLOBAL FACTORS")
print("=" * 80)

# For feature 0: expected global factor ~0.69
# Check if local parts * 0.69 match expected
print("\nFeature 0:")
print("  Expected values: 0.23, 0.46, 0.46, 0.23, 0.23, 0.23, 0.23, 0.46, 0.46")
print("  If gf=0.69:")
print("    0.23/0.69 = {:.4f} (local), 0.46/0.69 = {:.4f} (local)".format(0.23/0.69, 0.46/0.69))
print("  So local parts are 1/3 and 2/3")
print("  Samples 0,3,5,7,8 have local=1/3, samples 1,2,4,6 have local=2/3")
print("  Wait -- sample 6 class=3, so need local=1/3 but expected 0.23")
print()

# For feature 4: expected global factor
# sample 0 class=1, F=0.35; sample 3 class=2, F=0.53
# If local_part for class 1 sample = 1/3: gf = 0.35/(1/3) = 1.05 -- impossible
# If local_part = 2/3: gf = 0.35/(2/3) = 0.525
# For sample 3: if local_part = 1/3: gf = 0.53/(1/3) = 1.59 -- impossible
# For sample 3: if local_part = 2/3: gf = 0.53/(2/3) = 0.795
# These don't match -> different global factors?
# Or different local parts?

print("Feature 4 reverse engineering:")
for i in range(n_samples):
    exp_val = expected[i, 4]
    class_i = y[i]
    class_total = class_size[class_i]
    # Try all possible local parts: k/3 for k in 1,2,3
    for k in range(1, class_total + 1):
        local = k / class_total
        gf = exp_val / local
        if 0 < gf <= 1:
            print(f"  sample {i} (class {class_i}): F={exp_val}, "
                  f"local={k}/{class_total}={local:.4f}, gf={gf:.4f}, "
                  f"inv = (1-{gf:.4f})*72/2 = {(1-gf)*72/2:.1f}")

print("\n" + "=" * 80)
print("CHECKING: Does descending partition for feature 4 give different results?")
print("=" * 80)
# Feature 4 is clearly descending (high values for class 1, low for class 2/3)
# Expected direction should be descending
j = 4
inv_desc = compute_inversion_count_for_feature(X, y, j, ascending=False)
gf_desc = 1.0 - 2.0 * inv_desc / denom
print(f"Descending inv={inv_desc}, gf={gf_desc:.4f}")

partition_desc = mp.descending_partitions[j]
F_desc = np.zeros(n_samples)
for k, mmi in enumerate(partition_desc):
    mmi_y = y[mmi]
    unique_classes, inverse, counts = np.unique(mmi_y, return_inverse=True, return_counts=True)
    same_class_counts = counts[inverse]
    class_sizes_for_mmi = np.array([class_size[c] for c in unique_classes])
    local_parts = same_class_counts / class_sizes_for_mmi[inverse]
    F_desc[mmi] = local_parts * gf_desc
    print(f"  MMI[{k}]: indices={mmi}, y={mmi_y}")
    for ii, si in enumerate(mmi):
        print(f"    sample {si}: local={local_parts[ii]:.4f}, F={local_parts[ii]*gf_desc:.4f}")

print(f"\nF_desc for feature 4: {np.round(F_desc, 4)}")
print(f"Expected:             {expected[:, 4]}")

# --- Step 11: What if the inversion count should be computed differently? ---
# The paper's formula might define inversions as pairs (i,j) where x_i < x_j but d_i > d_j
# (strictly less in feature, strictly greater in decision)
# Let's compute this manually
print("\n" + "=" * 80)
print("MANUAL PAIR-WISE INVERSION COUNTS (strict feature ordering)")
print("=" * 80)

for j in [0, 1, 4, 7]:
    # Ascending: count pairs where x_i < x_j but y_i > y_j
    inv_strict_asc = 0
    inv_strict_desc = 0
    total_ordered_pairs = 0
    for i in range(n_samples):
        for jj in range(i+1, n_samples):
            # Ascending inversions: x_i < x_j but y_i > y_j, OR x_i > x_j but y_i < y_j
            if X[i, j] < X[jj, j]:
                total_ordered_pairs += 1
                if y[i] > y[jj]:
                    inv_strict_asc += 1
            elif X[i, j] > X[jj, j]:
                total_ordered_pairs += 1
                if y[i] < y[jj]:
                    inv_strict_asc += 1
            # For descending: x_i < x_j but y_i < y_j, OR x_i > x_j but y_i > y_j
            if X[i, j] < X[jj, j]:
                if y[i] < y[jj]:
                    inv_strict_desc += 1
            elif X[i, j] > X[jj, j]:
                if y[i] > y[jj]:
                    inv_strict_desc += 1

    gf_strict_asc = 1.0 - 2.0 * inv_strict_asc / denom
    gf_strict_desc = 1.0 - 2.0 * inv_strict_desc / denom

    impl_inv_asc = compute_inversion_count_for_feature(X, y, j, ascending=True)
    impl_inv_desc = compute_inversion_count_for_feature(X, y, j, ascending=False)

    print(f"Feature {j}:")
    print(f"  Strict asc inversions: {inv_strict_asc}, gf={gf_strict_asc:.4f} "
          f"(impl: inv={impl_inv_asc}, gf={1-2*impl_inv_asc/denom:.4f})")
    print(f"  Strict desc inversions: {inv_strict_desc}, gf={gf_strict_desc:.4f} "
          f"(impl: inv={impl_inv_desc}, gf={1-2*impl_inv_desc/denom:.4f})")

# --- Step 12: Check the descending lexsort tiebreaker ---
print("\n" + "=" * 80)
print("DESCENDING INVERSION: lexsort tiebreaker analysis")
print("=" * 80)

for j in [0, 4]:
    print(f"\nFeature {j}:")
    # The implementation uses: np.lexsort((-y, X[:, j]))
    # This sorts by X[:,j] ascending, then by -y ascending (i.e., y descending) as tiebreaker
    # Then counts inversions in -y[sorted]
    sorted_desc = np.lexsort((-y, X[:, j]))
    y_neg_sorted = (-y[sorted_desc]).astype(float)
    print(f"  Sorted indices: {sorted_desc}")
    print(f"  X values: {X[sorted_desc, j]}")
    print(f"  y values: {y[sorted_desc]}")
    print(f"  -y values: {y_neg_sorted}")

    # Manual inversion count of -y sorted
    inv_manual = 0
    for i in range(len(y_neg_sorted)):
        for jj in range(i+1, len(y_neg_sorted)):
            if y_neg_sorted[i] > y_neg_sorted[jj]:
                inv_manual += 1
    print(f"  Manual inversions in -y sorted: {inv_manual}")

# --- Final summary ---
print("\n" + "=" * 80)
print("SUMMARY OF ALL FEATURES: actual vs expected column sums")
print("=" * 80)
for j in range(n_features):
    print(f"  Feature {j}: actual_sum={np.sum(fd.matrix[:, j]):.4f}, "
          f"expected_sum={np.sum(expected[:, j]):.4f}, "
          f"direction={fd.monotone_directions[j]}")
