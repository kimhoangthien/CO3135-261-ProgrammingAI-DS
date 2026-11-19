"""Member 4 evaluation and calibration tests use synthetic data only."""
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from scripts.run_evaluation_explainability import main as workflow_main
from src.calibration import (build_calibrated_classifier, calibration_table,
                             evaluate_calibration, select_calibration_candidate)
from src.config import ROOT
from src.data import FEATURES, load_data
from src.evaluate import (bootstrap_confidence_intervals,
                          compare_prediction_frames,
                          cross_partition_duplicate_mask,
                          evaluate_probabilities,
                          paired_bootstrap_differences, prediction_frame,
                          validation_predictions)
from src.preprocessing import build_model_pipeline
from src.split import make_splits, save_splits


def test_extended_probability_metrics_preserve_existing_contract():
    metrics = evaluate_probabilities([0, 0, 1, 1], [.1, .4, .35, .8])
    assert metrics['probability_metrics']['roc_auc'] == .75
    assert metrics['probability_metrics']['pr_auc_average_precision'] == pytest.approx(5 / 6)
    assert metrics['probability_metrics']['brier_score'] == pytest.approx(.158125)
    assert metrics['probability_metrics']['log_loss'] > 0
    assert metrics['threshold_metrics']['confusion_matrix'] == [[2, 0], [1, 1]]


def test_prediction_frame_supports_contract_splits_and_validation_wrapper():
    frame = prediction_frame([4, 8], [0, 1], [.2, .9], 'example', 'test')
    assert frame['split'].tolist() == ['test', 'test']
    validation = validation_predictions([4, 8], [0, 1], [.2, .9], 'example')
    assert validation['split'].eq('validation').all()
    with pytest.raises(ValueError, match='Split'):
        prediction_frame([4, 8], [0, 1], [.2, .9], 'example', 'development')


def test_compare_prediction_frames_aligns_by_sample_id():
    first = validation_predictions([9, 3, 7], [1, 0, 1], [.8, .1, .4], 'first')
    second = validation_predictions([7, 9, 3], [1, 1, 0], [.7, .9, .2], 'second')
    result = compare_prediction_frames({'a': first, 'b': second})
    assert result['model_name'].tolist() == ['first', 'second']
    assert result['n_observations'].tolist() == [3, 3]
    misaligned = second.copy()
    misaligned.loc[0, 'y_true'] = 0
    with pytest.raises(ValueError, match='not aligned'):
        compare_prediction_frames({'a': first, 'b': misaligned})


def test_bootstrap_intervals_and_paired_differences_are_reproducible():
    y = np.array([0] * 16 + [1] * 4)
    weak = np.r_[np.linspace(.05, .95, 16), [.2, .4, .6, .8]]
    strong = np.r_[np.linspace(.01, .25, 16), np.linspace(.70, .95, 4)]
    first = bootstrap_confidence_intervals(
        y, strong, n_resamples=40, confidence_level=.9, seed=42,
    )
    second = bootstrap_confidence_intervals(
        y, strong, n_resamples=40, confidence_level=.9, seed=42,
    )
    pd.testing.assert_frame_equal(first, second)
    assert (first['ci_lower'] <= first['estimate']).all()
    assert (first['estimate'] <= first['ci_upper']).all()

    differences = paired_bootstrap_differences(
        y, weak, strong, n_resamples=40, confidence_level=.9, seed=42,
    ).set_index('metric')
    assert differences.loc['roc_auc', 'delta_model_b_minus_model_a'] > 0
    assert differences.loc['brier_score', 'delta_model_b_minus_model_a'] < 0


def test_cross_partition_duplicate_mask_handles_missing_values():
    train = pd.DataFrame({'a': [1, 2, np.nan], 'b': ['x', 'y', 'z']})
    evaluation = pd.DataFrame({
        'a': [2, np.nan, 3],
        'b': ['y', 'z', 'new'],
    }, index=[10, 11, 12])
    mask = cross_partition_duplicate_mask(train, evaluation, ['a', 'b'])
    assert mask.index.tolist() == [10, 11, 12]
    assert mask.tolist() == [True, True, False]


def test_calibration_table_counts_every_observation_and_reports_gaps():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([.05, .15, .25, .65, .75, .95])
    metrics, bins = evaluate_calibration(y, p, n_bins=3, strategy='quantile')
    assert bins['count'].sum() == len(y)
    assert len(bins) == 3
    assert bins['absolute_gap'].ge(0).all()
    assert 0 <= metrics['calibration_metrics']['expected_calibration_error'] <= 1
    uniform = calibration_table(y, p, n_bins=5, strategy='uniform')
    assert uniform['count'].sum() == len(y)


def test_calibration_handles_constant_probabilities():
    bins = calibration_table([0, 0, 1, 1], [.5, .5, .5, .5], n_bins=10)
    assert len(bins) == 1
    assert bins.iloc[0]['count'] == 4
    assert bins.iloc[0]['absolute_gap'] == 0


def test_training_fold_calibrator_clones_and_fits_shared_pipeline():
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=3,
        weights=[.75, .25], random_state=42,
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    base = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), scale=True,
    )
    calibrated = build_calibrated_classifier(base, method='sigmoid', cv=3, seed=42)
    calibrated.fit(X.iloc[:90], y[:90])
    probabilities = calibrated.predict_proba(X.iloc[90:])[:, 1]
    assert not hasattr(base, 'classes_')
    assert list(calibrated.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_select_calibration_candidate_uses_brier_then_log_loss():
    candidates = {
        'uncalibrated': {'probability_metrics': {'brier_score': .12, 'log_loss': .4}},
        'sigmoid': {'probability_metrics': {'brier_score': .10, 'log_loss': .5}},
        'isotonic': {'probability_metrics': {'brier_score': .10, 'log_loss': .3}},
    }
    assert select_calibration_candidate(candidates) == 'isotonic'


def test_member4_config_and_notebook_are_valid():
    with (ROOT / 'configs/evaluation.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    assert config['calibration']['methods'] == ['sigmoid', 'isotonic']
    assert config['explainability']['model'] in config['models']
    notebook = nbformat.read(ROOT / 'notebooks/04_evaluation.ipynb', as_version=4)
    nbformat.validate(notebook)


def test_member4_workflow_runs_end_to_end_without_reading_test(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(200, len(FEATURES))), columns=FEATURES,
    )
    frame['SeriousDlqin2yrs'] = np.tile([0] * 9 + [1], 20)
    data_path = tmp_path / 'cs-training.csv'
    frame.to_csv(data_path, index=False)
    frame = load_data(data_path)
    splits = make_splits(frame, seed=42)
    manifest = tmp_path / 'processed' / 'splits.json'
    save_splits(frame, splits, manifest, seed=42)

    train = frame.iloc[splits['train']]
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, random_state=42), scale=True,
    ).fit(train[FEATURES], train['SeriousDlqin2yrs'])
    input_artifacts = tmp_path / 'inputs'
    input_artifacts.mkdir()
    model_path = input_artifacts / 'logistic.joblib'
    joblib.dump(pipeline, model_path)

    config_path = tmp_path / 'default.yaml'
    config_path.write_text(yaml.safe_dump({
        'dataset_path': str(data_path),
        'target': 'SeriousDlqin2yrs',
        'random_seed': 42,
        'split_ratios': [.7, .15, .15],
        'outputs': {
            'processed': str(manifest.parent),
            'artifacts': str(tmp_path / 'artifacts'),
            'figures': str(tmp_path / 'figures'),
            'tables': str(tmp_path / 'tables'),
        },
    }), encoding='utf-8')
    evaluation_path = tmp_path / 'evaluation.yaml'
    evaluation_path.write_text(yaml.safe_dump({
        'threshold': .5,
        'models': {'logistic_tuned': str(model_path)},
        'calibration': {
            'methods': ['sigmoid', 'isotonic'], 'cv': 2,
            'n_bins': 5, 'strategy': 'quantile',
        },
        'explainability': {
            'model': 'logistic_tuned', 'background_size': 20,
            'global_sample_size': 10, 'local_cases': 3,
            'local_top_features': 5,
        },
    }), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'run_evaluation_explainability.py', '--config', str(config_path),
        '--evaluation-config', str(evaluation_path),
    ])
    workflow_main()

    tables = tmp_path / 'tables' / '04_evaluation'
    artifacts = tmp_path / 'artifacts' / '04_evaluation'
    summary = pd.read_csv(tables / 'evaluation_summary.csv')
    assert set(summary['calibration_method']) == {'uncalibrated', 'sigmoid', 'isotonic'}
    assert summary['selected_for_model'].sum() == 1
    assert (tables / 'bootstrap_confidence_intervals.csv').is_file()
    assert (tables / 'duplicate_sensitivity.csv').is_file()
    assert (tables / 'shap_global_importance.csv').is_file()
    assert (tables / 'shap_local_explanations.csv').is_file()
    assert (artifacts / 'logistic_tuned__sigmoid.joblib').is_file()
    metadata = yaml.safe_load((artifacts / 'configuration.yaml').read_text(encoding='utf-8'))
    assert metadata['evaluation_split'] == 'validation'
    assert metadata['test_evaluated'] is False
