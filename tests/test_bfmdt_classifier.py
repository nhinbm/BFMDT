import pytest
import numpy as np
from bfmdt.bfmdt_classifier import BFMDTClassifier


@pytest.fixture
def paper_dataset():
    """Dataset from Table I of the BFMDT paper (9 samples, 8 features, 3 classes)."""
    X = np.array([
        [0.1, 0.0, 0.3, 0.2, 1.0, 0.4, 0.6, 0.8],  # x1
        [0.3, 0.1, 0.6, 0.0, 0.6, 0.2, 0.1, 0.9],  # x2
        [0.3, 0.0, 0.5, 0.5, 0.9, 0.0, 0.7, 0.6],  # x3
        [0.1, 0.3, 0.3, 0.5, 0.1, 0.1, 0.8, 0.7],  # x4
        [0.3, 0.2, 0.7, 0.7, 0.0, 0.4, 0.4, 0.5],  # x5
        [0.0, 0.5, 0.1, 1.0, 0.1, 0.5, 0.1, 0.3],  # x6
        [0.6, 0.5, 0.1, 0.8, 0.6, 0.5, 0.0, 0.2],  # x7
        [0.1, 0.6, 1.0, 1.0, 0.2, 0.7, 0.3, 0.2],  # x8
        [0.1, 1.0, 1.0, 0.7, 0.1, 1.0, 1.0, 0.0],  # x9
    ])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    return X, y


def test_fit_returns_self(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.5, delta=0.01)
    result = clf.fit(X, y)
    assert result is clf


def test_predict_shape(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.5, delta=0.01)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_predict_proba_shape_and_sums(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.5, delta=0.01)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]),
                               atol=1e-10)


def test_predict_labels_in_original_space(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.5, delta=0.01)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert set(y_pred).issubset({1, 2, 3})


def test_training_accuracy_positive(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.5, delta=0.01)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    assert accuracy > 0.0


def test_fixed_sigma(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma=0.3, delta=0.01)
    clf.fit(X, y)
    assert clf.best_sigma == 0.3


def test_auto_sigma(paper_dataset):
    X, y = paper_dataset
    clf = BFMDTClassifier(sigma="auto", delta=0.01)
    clf.fit(X, y)
    assert clf.best_sigma is not None
    assert isinstance(clf.best_sigma, float)


def test_predict_before_fit_raises():
    clf = BFMDTClassifier()
    X_test = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    with pytest.raises(ValueError):
        clf.predict(X_test)
