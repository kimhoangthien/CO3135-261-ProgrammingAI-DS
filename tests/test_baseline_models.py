
import numpy as np
from sklearn.datasets import make_classification

from src.models.logistic import (
    build_logistic_baseline,
    tune_logistic,
)
from src.models.random_forest import (
    build_random_forest_baseline,
    tune_random_forest,
)

def make_test_data():
    """Generate synthetic binary classification data."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    return X, y


def test_logistic_baseline():
    X, y = make_test_data()

    model = build_logistic_baseline(
        seed=42,
        max_iter=2000,
        class_weight="balanced",
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (300, 2)
    assert list(model.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_logistic_tuning():
    X, y = make_test_data()

    search = tune_logistic(
        X,
        y,
        seed=42,
        cv=3,
    )

    assert search.best_estimator_ is not None
    assert "model__C" in search.best_params_
    assert 0 <= search.best_score_ <= 1

    probabilities = search.best_estimator_.predict_proba(X)

    assert probabilities.shape == (300, 2)
    assert list(search.best_estimator_.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_random_forest_baseline():
    X, y = make_test_data()

    model = build_random_forest_baseline(
        seed=42,
        n_estimators=20,
        max_depth=5,
        n_jobs=1,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (300, 2)
    assert list(model.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )

    assert model.named_steps["model"].n_estimators == 20


def test_random_forest_tuning():
    X, y = make_test_data()

    search = tune_random_forest(
        X,
        y,
        seed=42,
        cv=2,
    )

    assert search.best_estimator_ is not None
    assert "model__n_estimators" in search.best_params_
    assert "model__max_depth" in search.best_params_
    assert "model__min_samples_leaf" in search.best_params_
    assert "model__class_weight" in search.best_params_

    assert 0 <= search.best_score_ <= 1

    probabilities = search.best_estimator_.predict_proba(X)

    assert probabilities.shape == (300, 2)
    assert list(search.best_estimator_.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )
