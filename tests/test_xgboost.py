"""Synthetic fixtures validate XGBoost code only; they are not research results."""
import joblib
import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from src.config import ROOT, load_config
from src.data import FEATURES
from src.models.xgboost import (STRATEGIES, XGB_SEARCH_SPACE, build_xgboost_pipeline,
                                compute_scale_pos_weight, param_prefix, tune_xgboost)
from src.train import verify_preprocessing

FAST = {"n_estimators": 10, "max_depth": 2}     # keeps every test fast


@pytest.fixture
def data(frame):
    """Train/holdout with missing values, like the real MonthlyIncome."""
    X = frame[FEATURES].copy()
    X.loc[X.index[::7], "MonthlyIncome"] = np.nan
    y = frame["SeriousDlqin2yrs"]
    return X.iloc[:140], y.iloc[:140], X.iloc[140:]


def make(strategy, y_train, **params):
    """Build any strategy; 'weighted' gets its ratio from the TRAIN labels."""
    spw = compute_scale_pos_weight(y_train) if strategy == "weighted" else None
    return build_xgboost_pipeline(seed=42, strategy=strategy, scale_pos_weight=spw,
                                  **{**FAST, **params})


# 1. compute_scale_pos_weight
def test_scale_pos_weight():
    ratio = compute_scale_pos_weight(np.array([0, 0, 0, 1]))
    assert ratio == pytest.approx(3.0)
    assert isinstance(ratio, float)          # plain float, safe for yaml.safe_dump
    with pytest.raises(ValueError):
        compute_scale_pos_weight(np.array([0, 0, 0]))


# 2. invalid inputs
@pytest.mark.parametrize("spw", [None, 0, -1.0])
def test_weighted_requires_positive_ratio(spw):
    with pytest.raises(ValueError, match="scale_pos_weight"):
        build_xgboost_pipeline(seed=42, strategy="weighted", scale_pos_weight=spw)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        build_xgboost_pipeline(strategy="oversample")
    with pytest.raises(ValueError):
        param_prefix("oversample")


def test_weighted_ratio_reaches_classifier(data):
    _, y_train, _ = data
    pipeline = make("weighted", y_train)
    expected = compute_scale_pos_weight(y_train)
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == expected


# 3. structure + 4. contract (one test, run for each strategy)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_contract(strategy, data):
    X_train, y_train, X_hold = data
    pipeline = make(strategy, y_train).fit(X_train, y_train)
    assert [name for name, _ in pipeline.steps] == ["preprocessing", "model"]
    assert list(pipeline.classes_) == [0, 1]
    proba = pipeline.predict_proba(X_hold)
    assert proba.shape == (len(X_hold), 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    report = verify_preprocessing(pipeline, X_train, X_hold)
    assert report["training_statistics_verified"]


# 5. SMOTE only during fit, never on holdout
def test_smote_resamples_only_training(data, monkeypatch):
    X_train, y_train, X_hold = data
    calls = []
    original = SMOTE.fit_resample

    def spy(self, X, y):
        X_res, y_res = original(self, X, y)
        calls.append((len(X), len(X_res)))
        return X_res, y_res

    # Patch the class: sklearn clones steps, so an instance-level spy would miss calls.
    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    pipeline = make("smote", y_train).fit(X_train, y_train)
    assert len(calls) == 1
    rows_in, rows_out = calls[0]
    assert rows_in == len(X_train)
    assert rows_out > rows_in
    pipeline.predict_proba(X_hold)
    assert len(calls) == 1                   # prediction never resamples


@pytest.mark.parametrize("strategy", ["standard", "weighted"])
def test_no_smote_outside_smote_strategy(strategy, data, monkeypatch):
    X_train, y_train, _ = data
    calls = []
    monkeypatch.setattr(SMOTE, "fit_resample", lambda self, X, y: calls.append(1))
    make(strategy, y_train).fit(X_train, y_train)
    assert calls == []


# 6. persistence
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_reload_gives_identical_probabilities(strategy, data, tmp_path):
    X_train, y_train, X_hold = data
    pipeline = make(strategy, y_train).fit(X_train, y_train)
    path = tmp_path / f"xgboost_{strategy}.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict_proba(X_hold)[:, 1],
                               pipeline.predict_proba(X_hold)[:, 1], rtol=0, atol=0)


# 7. reproducibility
def test_same_seed_same_probabilities(data):
    X_train, y_train, X_hold = data
    first = make("smote", y_train, subsample=0.8).fit(X_train, y_train)
    second = make("smote", y_train, subsample=0.8).fit(X_train, y_train)
    np.testing.assert_array_equal(first.predict_proba(X_hold), second.predict_proba(X_hold))


# 8. tuning
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_tuning(strategy, frame):
    X, y = frame[FEATURES], frame["SeriousDlqin2yrs"]
    spw = compute_scale_pos_weight(y) if strategy == "weighted" else None
    space = {"n_estimators": [5, 10], "max_depth": [2, 3]}
    search = tune_xgboost(X, y, strategy=strategy, cv=3, n_iter=2,
                          search_space=space, scale_pos_weight=spw)
    prefix = param_prefix(strategy)
    assert set(search.best_params_) == {prefix + name for name in space}
    assert 0 <= search.best_score_ <= 1
    assert list(search.best_estimator_.classes_) == [0, 1]
    assert search.scoring == "average_precision"


# 9. config file stays consistent with the code
def test_config_matches_code():
    cfg = load_config(ROOT / "configs/xgboost.yaml")
    assert cfg["search_space"] == XGB_SEARCH_SPACE
    assert set(cfg["strategies"]) <= set(STRATEGIES)
    assert cfg["tuning"]["cv"] >= 2
    assert cfg["tuning"]["n_iter"] >= 1
    assert 0 <= cfg["threshold"] <= 1
    for name, values in cfg["search_space"].items():
        assert values and all(isinstance(v, (int, float)) for v in values), name
