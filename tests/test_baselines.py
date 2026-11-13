"""Synthetic fixtures validate code only; they are not research results."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.config import load_config
from src.data import FEATURES, load_data
from src.evaluate import PREDICTION_COLUMNS, evaluate_probabilities, validation_predictions
from src.split import load_splits, make_splits, save_splits
from src.train import run_baselines


def test_known_metrics():
    metrics = evaluate_probabilities([0, 0, 1, 1], [.1, .4, .35, .8])
    assert metrics['probability_metrics']['roc_auc'] == .75
    assert metrics['probability_metrics']['brier_score'] == pytest.approx(.158125)
    assert metrics['probability_metrics']['pr_auc_average_precision'] == pytest.approx(5/6)
    assert metrics['threshold_metrics']['confusion_matrix'] == [[2, 0], [1, 1]]
    assert metrics['threshold_metrics']['precision'] == 1
    assert metrics['threshold_metrics']['recall'] == .5
    assert metrics['threshold_metrics']['f1'] == pytest.approx(2/3)


@pytest.mark.parametrize('p', [[.1], [np.nan, .5], [-.1, .5], [.1, 1.1]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        evaluate_probabilities([0, 1], p)


def test_prediction_identifiers():
    with pytest.raises(ValueError, match='Sample IDs'):
        validation_predictions([1, 1], [0, 1], [.1, .9], 'example')


def test_baseline_workflow(frame, tmp_path, monkeypatch):
    path = tmp_path / 'unit-fixture.csv'
    frame.to_csv(path, index=False)
    cfg = load_config()
    cfg['dataset_path'] = str(path)
    cfg['outputs'] = {name: str(tmp_path / name) for name in cfg['outputs']}
    persisted_frame = load_data(path, cfg['target'])
    parts = make_splits(
        persisted_frame, cfg['target'], cfg['split_ratios'], cfg['random_seed'])
    manifest = tmp_path / 'processed/splits.json'
    save_splits(
        persisted_frame, parts, manifest, cfg['random_seed'], cfg['split_ratios'])
    fit_rows, predict_rows = [], []
    original_fit, original_predict = Pipeline.fit, Pipeline.predict_proba
    def fit(self, X, y, **kwargs):
        fit_rows.append(list(X.index))
        return original_fit(self, X, y, **kwargs)
    def predict(self, X, **kwargs):
        predict_rows.append(list(X.index))
        return original_predict(self, X, **kwargs)
    monkeypatch.setattr(Pipeline, 'fit', fit)
    monkeypatch.setattr(Pipeline, 'predict_proba', predict)
    metrics = run_baselines(cfg)
    parts = load_splits(pd.read_csv(path), manifest)
    original_manifest = manifest.read_bytes()
    assert all(ids == parts['train'] for ids in fit_rows)
    assert all(ids == parts['validation'] for ids in predict_rows)
    for name in ['dummy_prior', 'logistic_regression']:
        predictions = pd.read_csv(tmp_path / f'tables/baselines/{name}_validation_predictions.csv')
        assert list(predictions.columns) == PREDICTION_COLUMNS
        assert predictions.sample_id.tolist() == parts['validation']
        assert predictions.split.eq('validation').all()
        saved = joblib.load(tmp_path / f'artifacts/baselines/{name}.joblib')
        expected = frame.iloc[parts['train']][FEATURES].median().to_numpy()
        np.testing.assert_allclose(saved.named_steps['preprocessing'].named_steps['imputer'].statistics_, expected)
        assert metrics[name]['reload_verified']
        assert metrics[name]['preprocessing']['training_statistics_verified']
    assert metrics['dummy_prior']['probability_metrics']['roc_auc'] == .5
    mismatched_cfg = {**cfg, 'random_seed': cfg['random_seed'] + 1}
    with pytest.raises(ValueError, match='split seed differs'):
        run_baselines(mismatched_cfg)
    assert manifest.read_bytes() == original_manifest
    run_baselines(cfg)
    assert manifest.read_bytes() == original_manifest
    metadata = (tmp_path / 'artifacts/baselines/configuration.yaml').read_text()
    assert 'test_evaluated: false' in metadata
    assert json.loads((tmp_path / 'tables/baselines/metrics.json').read_text())


def test_missing_data_fails_before_artifacts(tmp_path):
    cfg = load_config()
    cfg['dataset_path'] = str(tmp_path / 'absent.csv')
    cfg['outputs']['artifacts'] = str(tmp_path / 'artifacts')
    with pytest.raises(FileNotFoundError):
        run_baselines(cfg)
    assert not (tmp_path / 'artifacts').exists()


def test_missing_shared_manifest_stops_without_creating_outputs(frame, tmp_path):
    path = tmp_path / 'unit-fixture.csv'
    frame.to_csv(path, index=False)
    cfg = load_config()
    cfg['dataset_path'] = str(path)
    cfg['outputs'] = {name: str(tmp_path / name) for name in cfg['outputs']}

    with pytest.raises(FileNotFoundError, match='Shared split manifest not found'):
        run_baselines(cfg)

    assert not (tmp_path / 'processed/splits.json').exists()
    assert not (tmp_path / 'artifacts').exists()
    assert not (tmp_path / 'tables').exists()


def test_future_estimator_preprocessing_contract(frame):
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from src.preprocessing import build_model_pipeline
    for estimator in [RandomForestClassifier(random_state=42), XGBClassifier(random_state=42)]:
        pipeline = clone(build_model_pipeline(estimator))
        assert callable(pipeline.fit) and callable(pipeline.predict_proba)
        transformed = pipeline.named_steps['preprocessing'].fit_transform(frame[FEATURES].iloc[:140])
        assert transformed.shape == (140, len(FEATURES))
        assert np.isfinite(transformed).all()
        # Estimators remain unfitted; no RF or XGBoost model is trained here.
        assert not hasattr(pipeline.named_steps['model'], 'classes_')
