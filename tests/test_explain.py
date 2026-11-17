"""Synthetic SHAP tests for shared Logistic Regression and tree pipelines."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluate import validation_predictions
from src.explain import (explain_pipeline, global_shap_importance,
                         local_shap_table, select_representative_cases)
from src.preprocessing import build_model_pipeline


def synthetic_frame():
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0,
        weights=[.7, .3], random_state=42,
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)]), y


def assert_explanation_contract(explanation, expected_rows, expected_features):
    assert explanation.values.shape == (expected_rows, expected_features)
    assert np.isfinite(explanation.values).all()
    importance = global_shap_importance(explanation)
    assert len(importance) == expected_features
    assert importance['mean_absolute_shap'].is_monotonic_decreasing
    local = local_shap_table(explanation, row=0, top_n=3)
    assert len(local) == 3
    assert list(local.columns) == [
        'feature', 'transformed_feature_value', 'shap_value', 'absolute_shap',
    ]


def test_logistic_pipeline_explains_positive_class():
    X, y = synthetic_frame()
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), scale=True,
    ).fit(X.iloc[:70], y[:70])
    explanation = explain_pipeline(
        pipeline, X.iloc[:70], X.iloc[70:75], background_size=20, seed=42,
    )
    assert_explanation_contract(explanation, 5, 5)


def test_random_forest_pipeline_normalizes_two_class_shap_output():
    X, y = synthetic_frame()
    pipeline = build_model_pipeline(
        RandomForestClassifier(n_estimators=15, max_depth=4, random_state=42,
                               n_jobs=1),
    ).fit(X.iloc[:70], y[:70])
    explanation = explain_pipeline(
        pipeline, X.iloc[:70], X.iloc[70:73], background_size=20, seed=42,
    )
    assert_explanation_contract(explanation, 3, 5)


def test_xgboost_pipeline_explains_configured_handoff_model():
    X, y = synthetic_frame()
    pipeline = build_model_pipeline(
        XGBClassifier(
            n_estimators=10, max_depth=3, learning_rate=.1,
            objective='binary:logistic', tree_method='hist', random_state=42,
            n_jobs=1,
        ),
    ).fit(X.iloc[:70], y[:70])
    explanation = explain_pipeline(
        pipeline, X.iloc[:70], X.iloc[70:73], background_size=20, seed=42,
    )
    assert_explanation_contract(explanation, 3, 5)


def test_representative_cases_are_validation_only_and_unique():
    predictions = validation_predictions(
        [10, 11, 12, 13, 14, 15],
        [1, 0, 1, 0, 1, 0],
        [.95, .80, .45, .10, .20, .55],
        'example',
    )
    cases = select_representative_cases(predictions, threshold=.5, max_cases=3)
    assert cases['case_type'].tolist() == [
        'highest_risk_positive', 'false_positive', 'false_negative',
    ]
    assert cases['sample_id'].is_unique
    assert cases['split'].eq('validation').all()
