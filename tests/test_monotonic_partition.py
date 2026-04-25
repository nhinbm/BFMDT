import pytest
import numpy as np

from bfmdt.monotonic_partition import MonotonicPartitioner


@pytest.fixture
def partitioned():
    """X=[3,1,4,2], y=[2,1,4,3] -- unsorted, has breaks in both directions."""
    X = np.array([[3.0], [1.0], [4.0], [2.0]])
    y = np.array([2, 1, 4, 3])
    return MonotonicPartitioner().fit(X, y)


# --- ascending: sort X asc -> y=[1,3,2,4], break at 3>2 -> AMMIs: [1,3], [0,2] ---


def test_ascending_mmi_count(partitioned):
    assert len(partitioned.ascending_partitions[0]) == 2


def test_ascending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][0], [1, 3])


def test_ascending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.ascending_partitions[0][1], [0, 2])


# --- descending: sort X asc -> y=[1,3,2,4], breaks at 1<3 and 2<4 ---


def test_descending_mmi_count(partitioned):
    assert len(partitioned.descending_partitions[0]) == 3


def test_descending_first_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][0], [1])


def test_descending_second_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][1], [3, 0])


def test_descending_third_mmi(partitioned):
    np.testing.assert_array_equal(partitioned.descending_partitions[0][2], [2])


# --- edge cases ---


def test_single_sample():
    mp = MonotonicPartitioner().fit(np.array([[5.0]]), np.array([1]))
    assert len(mp.ascending_partitions[0]) == 1


def test_tied_features_one_mmi_ascending():
    """Def. 7 cond. 2: samples sharing a feature value must be in one MMI."""
    mp = MonotonicPartitioner().fit(np.array([[1.0], [1.0], [1.0]]), np.array([3, 1, 2]))
    assert len(mp.ascending_partitions[0]) == 1


def test_tied_features_one_mmi_descending():
    """Def. 7 cond. 2: same-feature samples must share one DMMI even when decisions differ."""
    mp = MonotonicPartitioner().fit(np.array([[1.0], [1.0], [1.0]]), np.array([3, 1, 2]))
    assert len(mp.descending_partitions[0]) == 1


def test_multiple_features():
    """Partitions computed independently per feature."""
    X = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    mp = MonotonicPartitioner().fit(X, np.array([1, 2, 3]))
    assert len(mp.ascending_partitions) == 2


def test_original_indices_preserved():
    """MMIs contain original sample indices, not sorted positions."""
    mp = MonotonicPartitioner().fit(np.array([[3.0], [1.0], [2.0]]), np.array([3, 1, 2]))
    np.testing.assert_array_equal(mp.ascending_partitions[0][0], [1, 2, 0])


def test_fit_returns_self():
    mp = MonotonicPartitioner()
    assert mp.fit(np.array([[1.0], [2.0]]), np.array([1, 2])) is mp


# --- Def. 7 cond. 2 (convexity) regression tests ---


def test_empty_input_raises():
    """n_samples == 0 must error out, not return a phantom empty MMI."""
    with pytest.raises(ValueError, match="empty sample set"):
        MonotonicPartitioner().fit(np.zeros((0, 1)), np.array([], dtype=int))


def test_descending_tied_features_grouped_with_mixed_features():
    """Original C1 failing case: ties at feature 0.1 must not be split across MMIs in DMP."""
    X = np.array([[0.1], [0.1], [0.2]])
    y = np.array([1, 2, 2])
    mp = MonotonicPartitioner().fit(X, y)
    dmp = mp.descending_partitions[0]

    # Indices 0 and 1 share feature value 0.1 -> Def. 7 cond. 2 forces them together.
    mmi_of = {idx: i for i, mmi in enumerate(dmp) for idx in mmi.tolist()}
    assert mmi_of[0] == mmi_of[1], "samples 0 and 1 share feature 0.1 -> same DMMI required"


def test_def7_cond2_invariant_no_overlapping_ranges_descending():
    """For DMP, no two MMIs may share any feature value (overlapping inf/sup ranges)."""
    X = np.array([[0.1], [0.1], [0.2], [0.2], [0.3]])
    y = np.array([1, 2, 1, 3, 2])
    mp = MonotonicPartitioner().fit(X, y)

    for direction_partitions in (mp.ascending_partitions[0], mp.descending_partitions[0]):
        ranges = [(X[mmi, 0].min(), X[mmi, 0].max()) for mmi in direction_partitions]
        for i, (lo_i, hi_i) in enumerate(ranges):
            for j, (lo_j, hi_j) in enumerate(ranges):
                if i == j:
                    continue
                assert hi_i < lo_j or hi_j < lo_i, (
                    f"MMIs {i} (range [{lo_i},{hi_i}]) and {j} (range [{lo_j},{hi_j}]) "
                    "overlap -- violates Def. 7 cond. 2"
                )


def test_def7_cond2_random_invariant():
    """Randomized: every sample's feature value must lie in exactly its own MMI's range."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 30)
        X = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=(n, 1))
        y = rng.integers(1, 5, size=n)
        mp = MonotonicPartitioner().fit(X, y)

        for partitions in (mp.ascending_partitions[0], mp.descending_partitions[0]):
            covered = np.concatenate(partitions)
            assert sorted(covered.tolist()) == list(range(n)), "coverage broken"

            for mmi in partitions:
                lo, hi = X[mmi, 0].min(), X[mmi, 0].max()
                in_range = np.where((X[:, 0] >= lo) & (X[:, 0] <= hi))[0]
                assert set(in_range.tolist()) <= set(mmi.tolist()), (
                    f"Def. 7 cond. 2 violated: samples {set(in_range.tolist()) - set(mmi.tolist())} "
                    f"have feature in [{lo},{hi}] but are not in this MMI"
                )


def test_paper_table1_feature_d_partition_is_valid():
    """Paper Table 1 feature d: implementation should produce a partition that
    respects Def. 7 cond. 2, even though it may not match the paper's narrative
    Example 1 grouping (the paper's example is internally inconsistent)."""
    X = np.array([[0.2], [0.0], [0.5], [0.5], [0.7], [0.0], [0.8], [0.2], [0.7]])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    mp = MonotonicPartitioner().fit(X, y)

    amp = mp.ascending_partitions[0]
    covered = sorted(np.concatenate(amp).tolist())
    assert covered == list(range(9)), "AMP must cover all 9 samples exactly once"

    # samples 1 and 5 both have d=0.0 -> must share an MMI (Def. 7 cond. 2)
    mmi_of = {idx: i for i, mmi in enumerate(amp) for idx in mmi.tolist()}
    assert mmi_of[1] == mmi_of[5]
    # samples 0 and 7 both have d=0.2 -> must share an MMI
    assert mmi_of[0] == mmi_of[7]
    # samples 2 and 3 both have d=0.5 -> must share an MMI
    assert mmi_of[2] == mmi_of[3]
    # samples 4 and 8 both have d=0.7 -> must share an MMI
    assert mmi_of[4] == mmi_of[8]
