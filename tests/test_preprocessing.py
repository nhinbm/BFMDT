import numpy as np

from bfmdt.preprocessing import Preprocessor


# --- Preprocessor must handle non-float label dtypes ---


def test_fit_transform_accepts_string_labels():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    y = np.array(['M', 'B', 'M'], dtype=object)
    _, y_clean = Preprocessor().fit_transform(X, y)
    assert len(y_clean) == 3
    np.testing.assert_array_equal(y_clean, np.array(['M', 'B', 'M'], dtype=object))


def test_fit_transform_accepts_integer_labels():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    y = np.array([0, 1, 0], dtype=int)
    _, y_clean = Preprocessor().fit_transform(X, y)
    assert len(y_clean) == 3


def test_fit_transform_drops_samples_with_nan_labels():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    y = np.array([0.0, np.nan, 1.0])
    X_clean, y_clean = Preprocessor().fit_transform(X, y)
    assert len(y_clean) == 2
    assert X_clean.shape[0] == 2


# --- numeric strings must preserve numeric (monotonic) order ---


def test_numeric_strings_preserve_numeric_order():
    """['1.0', '10.0', '2.0'] must NOT be lex-sorted into [0, 2, 1]."""
    X = np.array([['1.0'], ['10.0'], ['2.0']], dtype=object)
    y = np.array([0, 2, 1])
    pre = Preprocessor()
    X_clean, _ = pre.fit_transform(X, y)

    assert X_clean[0, 0] == 0.0
    assert X_clean[1, 0] == 1.0
    assert 0.0 < X_clean[2, 0] < 1.0
    assert len(pre.categorical_cols_) == 0


def test_partial_numeric_coerced_not_encoded():
    """Columns with mostly numeric strings + non-numeric ('N/A') should be coerced, not encoded."""
    X = np.array([['1.0'], ['N/A'], ['2.0'], ['3.0']], dtype=object)
    y = np.array([0, 1, 2, 3])
    pre = Preprocessor()
    pre.fit_transform(X, y)
    assert len(pre.categorical_cols_) == 0


def test_truly_categorical_columns_are_encoded():
    X = np.array([['red'], ['blue'], ['green'], ['red']], dtype=object)
    y = np.array([0, 1, 2, 0])
    pre = Preprocessor()
    pre.fit_transform(X, y)
    assert pre.categorical_cols_ == [0]
    assert 0 in pre.label_mappings_


def test_numeric_dtype_skips_categorical_loop():
    """Pure-numeric arrays should not populate categorical_cols_."""
    X = np.random.rand(30, 50)
    y = np.random.randint(0, 3, 30)
    pre = Preprocessor()
    pre.fit_transform(X, y)
    assert pre.categorical_cols_ == []
    assert pre.label_mappings_ == {}


# --- transform() must not collapse categoricals on the test set ---


def test_transform_categorical_not_collapsed():
    """All M's and B's in test must NOT map to a single constant."""
    X_train = np.array(
        [['M', 1.0], ['B', 2.0], ['M', 3.0], ['B', 4.0]],
        dtype=object,
    )
    y_train = np.array([0, 1, 0, 1])
    pre = Preprocessor()
    pre.fit_transform(X_train, y_train)

    X_test = np.array([['M', 1.5], ['B', 2.5], ['M', 3.5]], dtype=object)
    X_test_clean = pre.transform(X_test)

    assert len(np.unique(X_test_clean[:, 0])) == 2


def test_transform_matches_fit_transform_on_same_data():
    """Transforming the training data should reproduce fit_transform output."""
    X = np.array(
        [['M', 1.0], ['B', 2.0], ['M', 3.0], ['B', 4.0]],
        dtype=object,
    )
    y = np.array([0, 1, 0, 1])
    pre = Preprocessor()
    X_train_clean, _ = pre.fit_transform(X, y)
    X_retransformed = pre.transform(X)
    np.testing.assert_array_almost_equal(X_train_clean, X_retransformed)


def test_transform_unseen_categorical_fallbacks_without_crash():
    X_train = np.array([['red'], ['blue'], ['red'], ['green']], dtype=object)
    y_train = np.array([0, 1, 0, 2])
    pre = Preprocessor()
    pre.fit_transform(X_train, y_train)

    X_test = np.array([['yellow'], ['red']], dtype=object)
    X_test_clean = pre.transform(X_test)
    assert X_test_clean.shape == (2, 1)


def test_transform_handles_numeric_dtype_when_train_was_object():
    """If a user passes floats where train had strings, fallback must not silently collapse."""
    X_train = np.array(
        [['M', 1.0], ['B', 2.0], ['M', 3.0], ['B', 4.0]],
        dtype=object,
    )
    y_train = np.array([0, 1, 0, 1])
    pre = Preprocessor()
    pre.fit_transform(X_train, y_train)

    X_test = np.array([[0.0, 1.5], [1.0, 2.5]], dtype=object)
    X_test_clean = pre.transform(X_test)
   
    assert X_test_clean.shape == (2, 2)


# --- Numeric pipeline correctness ---


def test_output_in_unit_range_fit():
    X = np.random.rand(50, 10) * 100
    y = np.random.randint(0, 3, 50)
    X_clean, _ = Preprocessor().fit_transform(X, y)
    assert X_clean.min() >= 0.0
    assert X_clean.max() <= 1.0


def test_output_in_unit_range_transform_with_out_of_range_test():
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 3, 50)
    pre = Preprocessor()
    pre.fit_transform(X_train, y_train)

    X_test = np.random.rand(20, 10) * 5 
    X_test_clean = pre.transform(X_test)
    assert X_test_clean.min() >= 0.0
    assert X_test_clean.max() <= 1.0


def test_constant_features_are_dropped():
    X = np.column_stack([np.random.rand(20), np.ones(20), np.random.rand(20)])
    y = np.random.randint(0, 2, 20)
    pre = Preprocessor()
    X_clean, _ = pre.fit_transform(X, y)
    assert X_clean.shape[1] == 2
    assert pre.valid_features_mask_.tolist() == [True, False, True]


def test_nan_values_imputed_with_mean():
    X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([0, 1, 2])
    pre = Preprocessor()
    pre.fit_transform(X, y)
    np.testing.assert_allclose(pre.feature_mean_[1], 5.0)


# --- fit()/transform() split + sklearn compatibility ---


def test_fit_returns_self():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    pre = Preprocessor()
    result = pre.fit(X, y)
    assert result is pre


def test_fit_then_transform_matches_fit_transform():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    pre1 = Preprocessor()
    X1, _ = pre1.fit_transform(X, y)

    pre2 = Preprocessor()
    pre2.fit(X, y)
    X2 = pre2.transform(X)

    np.testing.assert_array_almost_equal(X1, X2)


def test_fit_transform_without_y_returns_only_x():
    X = np.random.rand(10, 5)
    pre = Preprocessor()
    result = pre.fit_transform(X)
    assert isinstance(result, np.ndarray)


def test_sklearn_clone_works():
    from sklearn.base import clone
    pre = Preprocessor(nan_strategy='mean', drop_constant=False)
    cloned = clone(pre)
    assert cloned.nan_strategy == 'mean'
    assert cloned.drop_constant is False


def test_get_params_returns_init_params():
    pre = Preprocessor(handle_unknown='error', drop_constant=False)
    params = pre.get_params()
    assert params['handle_unknown'] == 'error'
    assert params['drop_constant'] is False


# --- Input validation (M4) ---


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError
    import pytest
    pre = Preprocessor()
    with pytest.raises(NotFittedError):
        pre.transform(np.random.rand(5, 3))


def test_transform_wrong_feature_count_raises():
    import pytest
    pre = Preprocessor()
    pre.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
    with pytest.raises(ValueError, match="features"):
        pre.transform(np.random.rand(5, 3))


def test_non_2d_input_raises():
    import pytest
    pre = Preprocessor()
    with pytest.raises(ValueError, match="2D"):
        pre.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))


def test_invalid_nan_strategy_raises():
    import pytest
    pre = Preprocessor(nan_strategy='median')
    with pytest.raises(ValueError, match="nan_strategy"):
        pre.fit(np.random.rand(10, 3), np.random.randint(0, 2, 10))


def test_handle_unknown_error_raises_on_unseen():
    import pytest
    X_train = np.array([['red'], ['blue'], ['red']], dtype=object)
    y_train = np.array([0, 1, 0])
    pre = Preprocessor(handle_unknown='error')
    pre.fit(X_train, y_train)

    X_test = np.array([['yellow']], dtype=object)
    with pytest.raises(ValueError, match="Unknown category"):
        pre.transform(X_test)


# --- Missing-value sentinels (? handled as NaN) ---


def test_question_mark_sentinel_treated_as_nan():
    """'?' should be converted to NaN and imputed (shared convention in UCI datasets)."""
    X = np.array([['1.0'], ['?'], ['2.0'], ['3.0']], dtype=object)
    y = np.array([0, 1, 2, 3])
    pre = Preprocessor()
    pre.fit_transform(X, y)
    assert pre.categorical_cols_ == []


# --- drop_constant=False keeps constant features ---


def test_drop_constant_false_keeps_constant_features():
    X = np.column_stack([np.random.rand(20), np.ones(20), np.random.rand(20)])
    y = np.random.randint(0, 2, 20)
    pre = Preprocessor(drop_constant=False)
    X_clean, _ = pre.fit_transform(X, y)
    assert X_clean.shape[1] == 3


# --- nominal_strategy='drop' ---


def test_nominal_drop_removes_pure_categorical_column():
    """Pure-categorical column should be dropped, numeric column preserved."""
    X = np.array(
        [['red', 1.0], ['blue', 2.0], ['green', 3.0], ['red', 4.0]],
        dtype=object,
    )
    y = np.array([0, 1, 2, 0])
    pre = Preprocessor(nominal_strategy='drop')
    X_clean, _ = pre.fit_transform(X, y)
    assert X_clean.shape[1] == 1
    assert pre.nominal_dropped_cols_ == [0]
    assert pre.label_mappings_ == {}
    assert pre.categorical_cols_ == []


def test_nominal_drop_removes_mixed_string_majority_column():
    """Mixed column with string_count > numeric_count should be dropped."""
    X = np.array(
        [['1.0', 1.0], ['foo', 2.0], ['bar', 3.0], ['baz', 4.0]],
        dtype=object,
    )
    y = np.array([0, 1, 2, 0])
    pre = Preprocessor(nominal_strategy='drop')
    X_clean, _ = pre.fit_transform(X, y)
    assert X_clean.shape[1] == 1
    assert pre.nominal_dropped_cols_ == [0]


def test_nominal_drop_keeps_mixed_numeric_majority_column():
    """Mixed column with numeric_count >= string_count should be kept + imputed."""
    X = np.array(
        [['1.0'], ['2.0'], ['foo'], ['3.0']],
        dtype=object,
    )
    y = np.array([0, 1, 2, 3])
    pre = Preprocessor(nominal_strategy='drop')
    X_clean, _ = pre.fit_transform(X, y)
    assert X_clean.shape[1] == 1
    assert pre.nominal_dropped_cols_ == []


def test_nominal_drop_raises_when_all_columns_dropped():
    import pytest
    X = np.array([['red'], ['blue'], ['green']], dtype=object)
    y = np.array([0, 1, 2])
    pre = Preprocessor(nominal_strategy='drop')
    with pytest.raises(ValueError, match="All features were dropped"):
        pre.fit_transform(X, y)


def test_nominal_drop_transform_reuses_learned_mask():
    """transform() must drop the same columns identified at fit time."""
    X_train = np.array(
        [['red', 1.0], ['blue', 2.0], ['green', 3.0], ['red', 4.0]],
        dtype=object,
    )
    y_train = np.array([0, 1, 2, 0])
    pre = Preprocessor(nominal_strategy='drop')
    pre.fit_transform(X_train, y_train)

    X_test = np.array([['yellow', 1.5], ['red', 2.5]], dtype=object)
    X_test_clean = pre.transform(X_test)
    assert X_test_clean.shape == (2, 1)


def test_nominal_encode_default_unchanged():
    """Default behavior must remain 'encode' (backward compatibility)."""
    X = np.array([['red'], ['blue'], ['green'], ['red']], dtype=object)
    y = np.array([0, 1, 2, 0])
    pre = Preprocessor()
    pre.fit_transform(X, y)
    assert pre.nominal_strategy == 'encode'
    assert pre.categorical_cols_ == [0]
    assert 0 in pre.label_mappings_
    assert pre.nominal_dropped_cols_ == []


def test_invalid_nominal_strategy_raises():
    import pytest
    pre = Preprocessor(nominal_strategy='onehot')
    with pytest.raises(ValueError, match="nominal_strategy"):
        pre.fit(np.random.rand(10, 3), np.random.randint(0, 2, 10))