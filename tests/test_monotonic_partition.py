import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner


# --- canonical AMP cases (DMP is symmetric, see test_dmp_is_amp_on_flipped_decision) ---


def test_perfect_monotonic_yields_single_granule():
    """y already monotonic w.r.t. X -> AMP collapses to one MMI covering U."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([1, 2, 3, 4]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 1
    np.testing.assert_array_equal(amp[0], [0, 1, 2, 3])


def test_constant_decision_yields_single_granule():
    """Constant y is non-decreasing AND non-increasing -> AMP and DMP both have one MMI."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([2, 2, 2, 2]),
    )
    assert len(mp.ascending_partitions[0]) == 1
    assert len(mp.descending_partitions[0]) == 1


def test_strictly_decreasing_decision_yields_singletons():
    """Every adjacent pair breaks ascending monotonicity -> n MMIs of size 1."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([4, 3, 2, 1]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 4
    for i, mmi in enumerate(amp):
        np.testing.assert_array_equal(mmi, [i])


def test_single_break_yields_two_granules():
    """One up-run followed by another -> two MMIs split at the downward step."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        np.array([1, 2, 3, 1, 2]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 2
    np.testing.assert_array_equal(amp[0], [0, 1, 2])
    np.testing.assert_array_equal(amp[1], [3, 4])


def test_multiple_breaks_yield_multiple_granules():
    """Repeated zig-zag in y produces one MMI per up-run."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        np.array([1, 2, 1, 2, 1, 2]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 3
    np.testing.assert_array_equal(amp[0], [0, 1])
    np.testing.assert_array_equal(amp[1], [2, 3])
    np.testing.assert_array_equal(amp[2], [4, 5])


# --- canonical DMP cases (constant-decision case is covered above) ---


def test_perfect_anti_monotonic_yields_single_granule_dmp():
    """y non-increasing w.r.t. X -> DMP collapses to one DMMI covering U."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([4, 3, 2, 1]),
    )
    dmp = mp.descending_partitions[0]
    assert len(dmp) == 1
    np.testing.assert_array_equal(dmp[0], [0, 1, 2, 3])


def test_strictly_increasing_decision_yields_singletons_dmp():
    """Every adjacent pair breaks descending monotonicity -> n DMMIs of size 1."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([1, 2, 3, 4]),
    )
    dmp = mp.descending_partitions[0]
    assert len(dmp) == 4
    for i, mmi in enumerate(dmp):
        np.testing.assert_array_equal(mmi, [i])


def test_single_break_yields_two_granules_dmp():
    """One down-run followed by another -> two DMMIs split at the upward step."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        np.array([5, 4, 3, 5, 4]),
    )
    dmp = mp.descending_partitions[0]
    assert len(dmp) == 2
    np.testing.assert_array_equal(dmp[0], [0, 1, 2])
    np.testing.assert_array_equal(dmp[1], [3, 4])


def test_multiple_breaks_yield_multiple_granules_dmp():
    """Repeated zig-zag in y produces one DMMI per down-run."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        np.array([2, 1, 2, 1, 2, 1]),
    )
    dmp = mp.descending_partitions[0]
    assert len(dmp) == 3
    np.testing.assert_array_equal(dmp[0], [0, 1])
    np.testing.assert_array_equal(dmp[1], [2, 3])
    np.testing.assert_array_equal(dmp[2], [4, 5])


# --- Algorithm 1 step 4: tie-break by ascending decision within feature ties ---


def test_tied_feature_sorted_by_decision_ascending():
    """Same-feature samples reordered by y asc -> y_sorted becomes monotonic, one MMI."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [1.0], [2.0]]),
        np.array([2, 1, 3]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 1
    np.testing.assert_array_equal(amp[0], [1, 0, 2])


def test_tied_feature_with_decreasing_decision_breaks():
    """Tie-break by y asc still leaves a downward step at the feature boundary -> split there."""
    mp = MonotonicPartitioner().fit(
        np.array([[1.0], [1.0], [2.0]]),
        np.array([3, 1, 2]),
    )
    amp = mp.ascending_partitions[0]
    assert len(amp) == 2
    np.testing.assert_array_equal(amp[0], [1, 0])
    np.testing.assert_array_equal(amp[1], [2])


# --- edge cases ---


def test_single_sample_yields_single_granule():
    mp = MonotonicPartitioner().fit(np.array([[5.0]]), np.array([3]))
    amp = mp.ascending_partitions[0]
    assert len(amp) == 1
    np.testing.assert_array_equal(amp[0], [0])


def test_empty_input_raises():
    """n_samples == 0 must error out, not return a phantom empty MMI."""
    with pytest.raises(ValueError, match="empty sample set"):
        MonotonicPartitioner().fit(np.zeros((0, 1)), np.array([], dtype=int))


def test_original_indices_preserved():
    """MMIs contain original sample indices, not sorted positions."""
    mp = MonotonicPartitioner().fit(
        np.array([[3.0], [1.0], [2.0]]),
        np.array([3, 1, 2]),
    )
    np.testing.assert_array_equal(mp.ascending_partitions[0][0], [1, 2, 0])


def test_fit_returns_self():
    mp = MonotonicPartitioner()
    assert mp.fit(np.array([[1.0], [2.0]]), np.array([1, 2])) is mp


# --- structural invariants (Def. 9 + Def. 7 cond. 2) ---


def test_partition_covers_U_and_pairwise_disjoint():
    """Def. 9: MMIs of one feature partition U exactly (∪Gt = U, Gt ∩ Gu = ∅)."""
    rng = np.random.default_rng(0)
    n = 30
    X = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=(n, 2))
    y = rng.integers(1, 5, size=n)
    mp = MonotonicPartitioner().fit(X, y)

    for j in range(X.shape[1]):
        for partitions in (mp.ascending_partitions[j], mp.descending_partitions[j]):
            covered = np.concatenate(partitions).tolist()
            assert sorted(covered) == list(range(n)), "coverage broken"
            assert len(set(covered)) == n, "MMIs are not pairwise disjoint"


def test_same_feature_value_implies_same_mmi():
    """Def. 7 cond. 2 (convexity): samples sharing a feature value cannot be split across MMIs."""
    rng = np.random.default_rng(1)
    n = 25
    X = rng.choice([0.1, 0.2, 0.3, 0.4], size=(n, 1))
    y = rng.integers(1, 5, size=n)
    mp = MonotonicPartitioner().fit(X, y)

    for partitions in (mp.ascending_partitions[0], mp.descending_partitions[0]):
        mmi_of = {idx: i for i, mmi in enumerate(partitions) for idx in mmi.tolist()}
        for v in np.unique(X[:, 0]):
            indices_with_v = np.where(X[:, 0] == v)[0]
            mmi_ids = {mmi_of[i] for i in indices_with_v}
            assert len(mmi_ids) == 1, f"feature value {v} split across MMIs {mmi_ids}"


# --- AMP / DMP symmetry ---


def test_dmp_is_amp_on_flipped_decision():
    """DMP(X, y) == AMP(X, M - y): flipping y turns descending runs into ascending ones."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 3, 2, 4, 1])

    dmp = MonotonicPartitioner().fit(X, y).descending_partitions[0]
    amp = MonotonicPartitioner().fit(X, y.max() - y).ascending_partitions[0]

    assert len(dmp) == len(amp)
    for d, a in zip(dmp, amp):
        np.testing.assert_array_equal(d, a)
