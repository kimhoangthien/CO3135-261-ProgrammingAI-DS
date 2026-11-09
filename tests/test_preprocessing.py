import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from src.data import FEATURES
from src.preprocessing import build_preprocessor, build_model_pipeline

def test_training_only_and_serialization(frame, tmp_path):
    train = frame[FEATURES].iloc[:140].copy()
    train.loc[0, 'MonthlyIncome'] = np.nan
    holdout = frame[FEATURES].iloc[140:].copy()
    holdout['MonthlyIncome'] = 1000000
    prep = build_preprocessor(scale=True)
    prep.fit(train)
    before = prep.named_steps['imputer'].statistics_.copy()
    assert np.isclose(before[4], np.nanmedian(train['MonthlyIncome']))
    prep.transform(holdout)
    np.testing.assert_array_equal(before, prep.named_steps['imputer'].statistics_)
    pipeline = build_model_pipeline(DummyClassifier(strategy='prior', random_state=42), scale=True)
    pipeline.fit(train, frame['SeriousDlqin2yrs'].iloc[:140])
    path = tmp_path / 'pipeline.joblib'
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(pipeline.predict_proba(holdout), joblib.load(path).predict_proba(holdout))


def test_tree_preprocessing_and_balancing(frame):
    import pytest
    from sklearn.tree import DecisionTreeClassifier
    estimator = DecisionTreeClassifier(random_state=42)
    pipeline = build_model_pipeline(estimator, balanced=True)
    pipeline.fit(frame[FEATURES], frame['SeriousDlqin2yrs'])
    assert pipeline.named_steps['model'].class_weight == 'balanced'
    assert pipeline.predict_proba(frame[FEATURES]).shape == (len(frame), 2)
    assert 'scaler' not in pipeline.named_steps['preprocessing'].named_steps
    with pytest.raises(ValueError, match='class_weight'):
        build_model_pipeline(DummyClassifier(), balanced=True)


def test_all_missing_feature_is_retained():
    X = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    prep = build_preprocessor(add_indicator=False)
    result = prep.fit_transform(X)
    assert result.shape == X.shape
    assert np.isfinite(result).all()
