"""Minimal Dummy and Logistic Regression baselines. Test rows are never evaluated."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import warnings

import joblib
import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from .config import ROOT, load_config, resolve
from .data import FEATURES, load_data
from .evaluate import evaluate_probabilities, validation_predictions
from .models.logistic import build_logistic_baseline
from .preprocessing import build_model_pipeline
from .split import (fingerprint, load_splits,
                    validate_split_configuration)


def shared_splits(frame, cfg):
    path = resolve(cfg['outputs']['processed']) / 'splits.json'
    if not path.exists():
        raise FileNotFoundError(
            f'Shared split manifest not found: {path}. Validate and create it '
            'with `python -m src.split` only after team approval.'
        )
    parts = load_splits(frame, path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_split_configuration(payload, cfg['random_seed'], cfg['split_ratios'])
    return parts, path


def verify_preprocessing(pipeline, X_train, X_validation):
    """Check learned statistics against training rows, without reading test features."""
    prep = pipeline.named_steps['preprocessing']
    assert list(prep.feature_names_in_) == FEATURES
    expected = X_train.median().fillna(0).to_numpy()
    np.testing.assert_allclose(prep.named_steps['imputer'].statistics_, expected)
    imputed_train = prep.named_steps['imputer'].transform(X_train)
    if 'scaler' in prep.named_steps:
        np.testing.assert_allclose(prep.named_steps['scaler'].mean_, imputed_train.mean(axis=0))
    train = prep.transform(X_train)
    validation = prep.transform(X_validation)
    if train.shape[1] != validation.shape[1] or not np.isfinite(validation).all():
        raise ValueError('Inconsistent or nonfinite transformed validation features')
    return {'train_rows': len(train), 'validation_rows': len(validation),
            'transformed_features': list(prep.get_feature_names_out()),
            'training_statistics_verified': True}


def run_baselines(cfg=None, baseline_cfg=None):
    cfg = cfg if cfg is not None else load_config()
    baseline_cfg = baseline_cfg if baseline_cfg is not None else load_config(ROOT / 'configs/baselines.yaml')
    frame = load_data(cfg['dataset_path'], cfg['target'])
    parts, manifest = shared_splits(frame, cfg)
    # No feature/label selection or prediction is performed on held-out test rows.
    train = frame.iloc[parts['train']]
    validation = frame.iloc[parts['validation']]
    X_train, y_train = train[FEATURES], train[cfg['target']]
    X_validation, y_validation = validation[FEATURES], validation[cfg['target']]
    models = {
        'dummy_prior': build_model_pipeline(DummyClassifier(random_state=cfg['random_seed'], **baseline_cfg['dummy'])),
        'logistic_regression': build_logistic_baseline(cfg['random_seed'], **baseline_cfg['logistic']),
    }
    artifacts = resolve(cfg['outputs']['artifacts']) / 'baselines'
    tables = resolve(cfg['outputs']['tables']) / 'baselines'
    artifacts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metrics = {}
    parameters = {}
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        if list(pipeline.classes_) != [0, 1]:
            raise ValueError('Expected class ordering [0, 1]')
        preprocessing = verify_preprocessing(pipeline, X_train, X_validation)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        path = artifacts / f'{name}.joblib'
        joblib.dump(pipeline, path)
        np.testing.assert_allclose(joblib.load(path).predict_proba(X_validation)[:, 1],
                                   probabilities, rtol=1e-12, atol=1e-12)
        prediction = validation_predictions(parts['validation'], y_validation, probabilities, name)
        prediction.to_csv(tables / f'{name}_validation_predictions.csv', index=False)
        metrics[name] = dict(evaluate_probabilities(y_validation, probabilities, baseline_cfg['threshold']),
                             preprocessing=preprocessing, reload_verified=True)
        parameters[name] = pipeline.named_steps['model'].get_params(deep=False)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    metadata = dict(configuration=cfg, baselines=baseline_cfg, estimator_parameters=parameters,
                    seed=cfg['random_seed'], features=FEATURES, dataset_fingerprint=fingerprint(frame),
                    git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                    python=platform.python_version(), positive_class='SeriousDlqin2yrs=1: serious delinquency within two years',
                    evaluation_split='validation', test_evaluated=False)
    (artifacts / 'configuration.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')
    (artifacts / 'splits.json').write_bytes(manifest.read_bytes())
    versions = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (artifacts / 'dependencies.txt').write_text('\n'.join(versions) + '\n', encoding='utf-8')
    (tables / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--baseline-config', default=str(ROOT / 'configs/baselines.yaml'))
    args = parser.parse_args()
    try:
        metrics = run_baselines(load_config(args.config), load_config(args.baseline_config))
    except (OSError, ValueError, ConvergenceWarning) as error:
        parser.exit(1, f'Baseline workflow stopped: {error}\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
