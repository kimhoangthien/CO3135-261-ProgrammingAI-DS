"""Member 4 validation-only evaluation, calibration and SHAP workflow."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.calibration import (build_calibrated_classifier, evaluate_calibration,  # noqa: E402
                             select_calibration_candidate)
from src.config import ROOT, load_config, resolve  # noqa: E402
from src.data import FEATURES, load_data  # noqa: E402
from src.evaluate import (bootstrap_confidence_intervals,  # noqa: E402
                          compare_prediction_frames,
                          cross_partition_duplicate_mask,
                          paired_bootstrap_differences,
                          probability_metric_values, validation_predictions)
from src.explain import (explain_pipeline, global_shap_importance,  # noqa: E402
                         local_shap_table, select_representative_cases)
from src.split import fingerprint  # noqa: E402
from src.train import shared_splits  # noqa: E402


def git_state():
    def run(*command):
        result = subprocess.run(['git', *command], cwd=ROOT, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = run('status', '--porcelain')
    return {
        'git_revision': run('rev-parse', 'HEAD'),
        'git_uncommitted_changes': bool(status) if status is not None else None,
    }


def load_trusted_models(model_paths, selected_names):
    """Load only configured local project artifacts and validate their contract."""
    models = {}
    missing = []
    for name in selected_names:
        path = resolve(model_paths[name])
        if not path.is_file():
            missing.append(str(path))
            continue
        model = joblib.load(path)
        if not hasattr(model, 'predict_proba') or list(getattr(model, 'classes_', [])) != [0, 1]:
            raise ValueError(f'{name}: artifact does not satisfy the binary model contract')
        models[name] = model
    if missing:
        formatted = '\n- '.join(missing)
        raise FileNotFoundError(
            'Required trusted model artifacts are missing. Regenerate or obtain '
            f'the Member 2/3 handoff files:\n- {formatted}'
        )
    return models


def summary_row(model_name, calibration_method, selected, metrics):
    probability = metrics['probability_metrics']
    decision = metrics['threshold_metrics']
    reliability = metrics['calibration_metrics']
    (tn, fp), (fn, tp) = decision['confusion_matrix']
    return {
        'model_name': model_name,
        'calibration_method': calibration_method,
        'selected_for_model': selected,
        **probability,
        'expected_calibration_error': reliability['expected_calibration_error'],
        'maximum_calibration_error': reliability['maximum_calibration_error'],
        'threshold': decision['threshold'],
        'precision': decision['precision'],
        'recall': decision['recall'],
        'f1': decision['f1'],
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def uncertainty_outputs(selected_predictions, uncertainty_cfg, tables_dir, seed):
    """Save per-model intervals and paired validation comparisons."""
    compare_prediction_frames(selected_predictions)
    interval_frames = []
    ordered = {}
    for base_name, frame in selected_predictions.items():
        frame = frame.sort_values('sample_id').reset_index(drop=True)
        candidate_name = frame['model_name'].iloc[0]
        intervals = bootstrap_confidence_intervals(
            frame['y_true'], frame['y_pred_proba'],
            n_resamples=uncertainty_cfg['n_resamples'],
            confidence_level=uncertainty_cfg['confidence_level'], seed=seed,
        )
        intervals.insert(0, 'candidate_name', candidate_name)
        intervals.insert(0, 'base_model', base_name)
        interval_frames.append(intervals)
        ordered[base_name] = frame
    confidence_intervals = pd.concat(interval_frames, ignore_index=True)
    confidence_intervals.to_csv(
        tables_dir / 'bootstrap_confidence_intervals.csv', index=False,
    )

    pair_frames = []
    for (name_a, frame_a), (name_b, frame_b) in combinations(ordered.items(), 2):
        differences = paired_bootstrap_differences(
            frame_a['y_true'], frame_a['y_pred_proba'], frame_b['y_pred_proba'],
            n_resamples=uncertainty_cfg['n_resamples'],
            confidence_level=uncertainty_cfg['confidence_level'], seed=seed,
        )
        differences.insert(0, 'candidate_b', frame_b['model_name'].iloc[0])
        differences.insert(0, 'candidate_a', frame_a['model_name'].iloc[0])
        differences.insert(0, 'base_model_b', name_b)
        differences.insert(0, 'base_model_a', name_a)
        pair_frames.append(differences)
    pairwise = (pd.concat(pair_frames, ignore_index=True) if pair_frames
                else pd.DataFrame())
    pairwise.to_csv(tables_dir / 'paired_bootstrap_differences.csv', index=False)
    return confidence_intervals, pairwise


def duplicate_sensitivity_outputs(train, validation, parts, selected_predictions,
                                  target, tables_dir):
    """Compare metrics before/after excluding exact train-overlap validation rows."""
    columns = [*FEATURES, target]
    overlap_mask = cross_partition_duplicate_mask(train, validation, columns)
    validation_ids = np.asarray(parts['validation'], dtype=int)
    overlap_ids = validation_ids[overlap_mask.to_numpy()]
    overlap_rows = pd.DataFrame({
        'sample_id': overlap_ids,
        'y_true': validation.loc[overlap_mask, target].astype(int).to_numpy(),
    })
    overlap_rows.to_csv(tables_dir / 'validation_duplicates_in_train.csv', index=False)
    rows = []
    for base_name, frame in selected_predictions.items():
        subsets = {
            'all_validation': frame,
            'excluding_exact_train_duplicates': frame[~frame['sample_id'].isin(overlap_ids)],
        }
        for subset_name, subset in subsets.items():
            if set(subset['y_true'].unique()) == {0, 1}:
                metrics = probability_metric_values(subset['y_true'], subset['y_pred_proba'])
            else:
                metrics = {metric: np.nan for metric in
                           ['roc_auc', 'pr_auc_average_precision', 'brier_score', 'log_loss']}
            rows.append({
                'base_model': base_name,
                'candidate_name': frame['model_name'].iloc[0],
                'subset': subset_name,
                'n_observations': len(subset),
                'n_positive': int(subset['y_true'].sum()),
                **metrics,
            })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(tables_dir / 'duplicate_sensitivity.csv', index=False)
    summary = {
        'comparison_columns': columns,
        'training_rows': len(train),
        'validation_rows': len(validation),
        'validation_rows_with_exact_train_duplicate': int(overlap_mask.sum()),
        'validation_overlap_rate': float(overlap_mask.mean()),
        'overlap_positive_rows': int(validation.loc[overlap_mask, target].sum()),
        'shared_split_changed': False,
    }
    (tables_dir / 'duplicate_overlap_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    return summary, sensitivity


def save_shap_outputs(model_name, pipeline, frame, parts, X_train, y_validation,
                      X_validation, threshold, explain_cfg, tables_dir):
    probabilities = pipeline.predict_proba(X_validation)[:, 1]
    predictions = validation_predictions(
        parts['validation'], y_validation, probabilities, model_name,
    )
    cases = select_representative_cases(
        predictions, threshold=threshold, max_cases=explain_cfg['local_cases'],
    )
    rng = np.random.default_rng(42)
    size = min(explain_cfg['global_sample_size'], len(X_validation))
    positions = np.sort(rng.choice(len(X_validation), size=size, replace=False))
    global_explanation = explain_pipeline(
        pipeline, X_train, X_validation.iloc[positions],
        background_size=explain_cfg['background_size'], seed=42,
    )
    global_shap_importance(global_explanation).to_csv(
        tables_dir / 'shap_global_importance.csv', index=False,
    )
    case_features = frame.iloc[cases['sample_id'].astype(int)][FEATURES]
    local_explanation = explain_pipeline(
        pipeline, X_train, case_features,
        background_size=explain_cfg['background_size'], seed=42,
    )
    base_values = np.asarray(local_explanation.base_values)
    local_rows = []
    for position, case in cases.reset_index(drop=True).iterrows():
        local = local_shap_table(
            local_explanation, row=position,
            top_n=explain_cfg['local_top_features'],
        )
        local.insert(0, 'rank', np.arange(1, len(local) + 1))
        local.insert(0, 'predicted_probability', case['y_pred_proba'])
        local.insert(0, 'y_true', case['y_true'])
        local.insert(0, 'sample_id', case['sample_id'])
        local.insert(0, 'case_type', case['case_type'])
        local['base_value'] = float(base_values if base_values.ndim == 0
                                    else base_values[position])
        local_rows.append(local)
    cases.to_csv(tables_dir / 'shap_representative_cases.csv', index=False)
    pd.concat(local_rows, ignore_index=True).to_csv(
        tables_dir / 'shap_local_explanations.csv', index=False,
    )
    return {
        'model_name': model_name,
        'background_source': 'training',
        'background_size': min(explain_cfg['background_size'], len(X_train)),
        'global_validation_sample_size': size,
        'local_sample_ids': cases['sample_id'].astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--evaluation-config', default=str(ROOT / 'configs/evaluation.yaml'))
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--skip-shap', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = load_config(args.evaluation_config)
    selected_names = args.models or list(ecfg['models'])
    unknown = set(selected_names) - set(ecfg['models'])
    if unknown:
        parser.error(f'Unknown configured models: {sorted(unknown)}')

    frame = load_data(cfg['dataset_path'], cfg['target'])
    parts, manifest_path = shared_splits(frame, cfg)
    train = frame.iloc[parts['train']]
    validation = frame.iloc[parts['validation']]
    X_train, y_train = train[FEATURES], train[cfg['target']]
    X_validation = validation[FEATURES]
    y_validation = validation[cfg['target']]
    models = load_trusted_models(ecfg['models'], selected_names)

    artifacts_dir = resolve(cfg['outputs']['artifacts']) / '04_evaluation'
    tables_dir = resolve(cfg['outputs']['tables']) / '04_evaluation'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    threshold = ecfg['threshold']
    calibration_cfg = ecfg['calibration']
    all_metrics = {}
    selections = {}
    summary = []
    selected_predictions = {}
    for model_name, pipeline in models.items():
        candidates = {'uncalibrated': pipeline}
        for method in calibration_cfg['methods']:
            calibrated = build_calibrated_classifier(
                pipeline, method=method, cv=calibration_cfg['cv'],
                seed=cfg['random_seed'],
            )
            calibrated.fit(X_train, y_train)
            if list(calibrated.classes_) != [0, 1]:
                raise ValueError(f'{model_name}__{method}: expected classes [0, 1]')
            path = artifacts_dir / f'{model_name}__{method}.joblib'
            joblib.dump(calibrated, path)
            candidates[method] = calibrated

        model_metrics = {}
        candidate_predictions = {}
        for method, candidate in candidates.items():
            candidate_name = model_name if method == 'uncalibrated' else f'{model_name}__{method}'
            probabilities = candidate.predict_proba(X_validation)[:, 1]
            metrics, bins = evaluate_calibration(
                y_validation, probabilities,
                n_bins=calibration_cfg['n_bins'],
                strategy=calibration_cfg['strategy'],
            )
            model_metrics[method] = metrics
            prediction = validation_predictions(
                parts['validation'], y_validation, probabilities, candidate_name,
            )
            prediction.to_csv(
                tables_dir / f'{candidate_name}_validation_predictions.csv', index=False,
            )
            candidate_predictions[method] = prediction
            bins.insert(0, 'calibration_method', method)
            bins.insert(0, 'model_name', model_name)
            bins.to_csv(tables_dir / f'{candidate_name}_calibration_bins.csv', index=False)

        selected = select_calibration_candidate(model_metrics)
        selections[model_name] = selected
        all_metrics[model_name] = model_metrics
        selected_predictions[model_name] = candidate_predictions[selected]
        for method, metrics in model_metrics.items():
            summary.append(summary_row(model_name, method, method == selected, metrics))

    pd.DataFrame(summary).to_csv(tables_dir / 'evaluation_summary.csv', index=False)
    (tables_dir / 'metrics.json').write_text(
        json.dumps(all_metrics, indent=2, allow_nan=False), encoding='utf-8',
    )

    uncertainty_cfg = ecfg.get('uncertainty', {
        'n_resamples': 1000,
        'confidence_level': .95,
    })
    confidence_intervals, pairwise = uncertainty_outputs(
        selected_predictions, uncertainty_cfg, tables_dir, cfg['random_seed'],
    )
    duplicate_summary, duplicate_sensitivity = duplicate_sensitivity_outputs(
        train, validation, parts, selected_predictions, cfg['target'], tables_dir,
    )

    shap_metadata = None
    if not args.skip_shap:
        shap_name = ecfg['explainability']['model']
        if shap_name not in models:
            raise ValueError(
                f'SHAP model {shap_name!r} was not loaded; include it in --models '
                'or pass --skip-shap.'
            )
        shap_metadata = save_shap_outputs(
            shap_name, models[shap_name], frame, parts, X_train, y_validation,
            X_validation, threshold, ecfg['explainability'], tables_dir,
        )

    metadata = {
        'configuration': cfg,
        'evaluation_configuration': ecfg,
        'selected_calibration_by_model': selections,
        'selection_metric': 'validation_brier_score_then_log_loss',
        'uncertainty': {
            **uncertainty_cfg,
            'method': 'class-stratified_percentile_bootstrap',
            'paired_model_comparisons': len(pairwise) > 0,
        },
        'duplicate_sensitivity': duplicate_summary,
        'seed': cfg['random_seed'],
        'features': FEATURES,
        'dataset_fingerprint': fingerprint(frame),
        **git_state(),
        'python': platform.python_version(),
        'library_versions': {
            package: importlib.metadata.version(package)
            for package in ['numpy', 'pandas', 'scikit-learn', 'shap']
        },
        'evaluation_split': 'validation',
        'test_evaluated': False,
        'shap': shap_metadata,
    }
    (artifacts_dir / 'configuration.yaml').write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8',
    )
    (artifacts_dir / 'splits.json').write_bytes(manifest_path.read_bytes())
    versions = sorted(
        f'{distribution.metadata["Name"]}=={distribution.version}'
        for distribution in importlib.metadata.distributions()
    )
    (artifacts_dir / 'dependencies.txt').write_text(
        '\n'.join(versions) + '\n', encoding='utf-8',
    )
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print('\nBootstrap confidence intervals:')
    print(confidence_intervals.round(4).to_string(index=False))
    if not pairwise.empty:
        print('\nPaired bootstrap differences:')
        print(pairwise.round(4).to_string(index=False))
    print('\nDuplicate sensitivity:')
    print(duplicate_sensitivity.round(4).to_string(index=False))
    print(f'Calibration selections: {selections}')
    print(f'Outputs: {artifacts_dir} and {tables_dir}')


if __name__ == '__main__':
    main()
