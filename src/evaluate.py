"""Shared probability, threshold and model-comparison evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

PREDICTION_COLUMNS = ['sample_id', 'y_true', 'y_pred_proba', 'model_name', 'split']
VALID_SPLITS = {'train', 'validation', 'test'}


def validate_binary_probabilities(y_true, probabilities):
    """Return aligned arrays after validating the shared binary contract."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError(
            'Expected aligned binary labels with both classes and finite '
            'probabilities in [0,1]'
        )
    return y.astype(int, copy=False), p


def probability_metric_values(y_true, probabilities):
    """Calculate ranking and proper probability scores without a threshold."""
    y, p = validate_binary_probabilities(y_true, probabilities)
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc_average_precision': float(average_precision_score(y, p)),
        'brier_score': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
    }


def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    """PR-AUC here means average precision (step-weighted area, not trapezoidal)."""
    y, p = validate_binary_probabilities(y_true, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    predicted = p >= threshold
    return {
        'probability_metrics': {
            **probability_metric_values(y, p),
        },
        'threshold_metrics': {
            'threshold': float(threshold),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        },
    }


def prediction_frame(sample_ids, y_true, probabilities, model_name, split):
    """Build one prediction artifact while enforcing the shared schema."""
    y, p = validate_binary_probabilities(y_true, probabilities)
    ids = np.asarray(sample_ids)
    if (len(ids) != len(y_true) or ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids)) != len(ids) or np.any(ids < 0)):
        raise ValueError('Sample IDs must be unique, nonnegative integer row positions')
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('Model name must be a nonempty string')
    if split not in VALID_SPLITS:
        raise ValueError(f'Split must be one of {sorted(VALID_SPLITS)}')
    return pd.DataFrame(dict(sample_id=ids, y_true=y, y_pred_proba=p,
                             model_name=model_name, split=split),
                        columns=PREDICTION_COLUMNS)


def validation_predictions(sample_ids, y_true, probabilities, model_name):
    """Backward-compatible constructor for validation prediction artifacts."""
    return prediction_frame(sample_ids, y_true, probabilities, model_name,
                            split='validation')


def compare_prediction_frames(frames, threshold=0.5, expected_split='validation'):
    """Evaluate aligned prediction artifacts and return one row per model.

    Alignment is checked by ``sample_id`` rather than file row order. This keeps
    comparisons paired while allowing contributors to export rows differently.
    """
    if not frames:
        raise ValueError('Provide at least one prediction frame')
    rows = []
    reference = None
    for fallback_name, frame in frames.items():
        if list(frame.columns) != PREDICTION_COLUMNS:
            raise ValueError(f'{fallback_name}: prediction schema differs from contract')
        if frame.empty or frame['sample_id'].duplicated().any():
            raise ValueError(f'{fallback_name}: sample IDs must be nonempty and unique')
        if not frame['split'].eq(expected_split).all():
            raise ValueError(f'{fallback_name}: expected only {expected_split} predictions')
        names = frame['model_name'].unique()
        if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
            raise ValueError(f'{fallback_name}: expected one nonempty model name')
        ordered = frame.sort_values('sample_id').reset_index(drop=True)
        current = ordered[['sample_id', 'y_true']]
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError(f'{fallback_name}: predictions are not aligned to the same observations')
        metrics = evaluate_probabilities(ordered['y_true'], ordered['y_pred_proba'], threshold)
        probability = metrics['probability_metrics']
        decision = metrics['threshold_metrics']
        (tn, fp), (fn, tp) = decision['confusion_matrix']
        rows.append({
            'model_name': names[0],
            'split': expected_split,
            'n_observations': len(ordered),
            **probability,
            'threshold': decision['threshold'],
            'precision': decision['precision'],
            'recall': decision['recall'],
            'f1': decision['f1'],
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        })
    return pd.DataFrame(rows)


def _validate_bootstrap_settings(n_resamples, confidence_level):
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError('n_resamples must be a positive integer')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must be strictly between 0 and 1')


def _stratified_bootstrap_indices(y, rng):
    """Resample each class separately so every replicate has both classes."""
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    return np.concatenate([
        rng.choice(negative, size=len(negative), replace=True),
        rng.choice(positive, size=len(positive), replace=True),
    ])


def bootstrap_confidence_intervals(y_true, probabilities, n_resamples=1000,
                                   confidence_level=.95, seed=42):
    """Return percentile intervals for validation probability metrics."""
    y, p = validate_binary_probabilities(y_true, probabilities)
    _validate_bootstrap_settings(n_resamples, confidence_level)
    point = probability_metric_values(y, p)
    samples = {metric: np.empty(n_resamples) for metric in point}
    rng = np.random.default_rng(seed)
    for iteration in range(n_resamples):
        indices = _stratified_bootstrap_indices(y, rng)
        values = probability_metric_values(y[indices], p[indices])
        for metric, value in values.items():
            samples[metric][iteration] = value
    alpha = (1 - confidence_level) / 2
    rows = []
    for metric, estimate in point.items():
        lower, upper = np.quantile(samples[metric], [alpha, 1 - alpha])
        rows.append({
            'metric': metric,
            'estimate': estimate,
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'confidence_level': confidence_level,
            'n_resamples': n_resamples,
        })
    return pd.DataFrame(rows)


def paired_bootstrap_differences(y_true, probabilities_a, probabilities_b,
                                 n_resamples=1000, confidence_level=.95,
                                 seed=42):
    """Compare two aligned models using identical stratified bootstrap rows.

    Deltas are always ``model_b - model_a``. Positive deltas favor model B for
    ROC-AUC/AP; negative deltas favor model B for Brier score/log loss.
    """
    y, first = validate_binary_probabilities(y_true, probabilities_a)
    second_y, second = validate_binary_probabilities(y_true, probabilities_b)
    if not np.array_equal(y, second_y):
        raise ValueError('Paired comparison labels are not aligned')
    _validate_bootstrap_settings(n_resamples, confidence_level)
    first_point = probability_metric_values(y, first)
    second_point = probability_metric_values(y, second)
    samples = {metric: np.empty(n_resamples) for metric in first_point}
    rng = np.random.default_rng(seed)
    for iteration in range(n_resamples):
        indices = _stratified_bootstrap_indices(y, rng)
        first_values = probability_metric_values(y[indices], first[indices])
        second_values = probability_metric_values(y[indices], second[indices])
        for metric in samples:
            samples[metric][iteration] = second_values[metric] - first_values[metric]
    alpha = (1 - confidence_level) / 2
    rows = []
    for metric in first_point:
        deltas = samples[metric]
        lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
        higher_is_better = metric in {'roc_auc', 'pr_auc_average_precision'}
        probability_b_better = (np.mean(deltas > 0) if higher_is_better
                                else np.mean(deltas < 0))
        rows.append({
            'metric': metric,
            'delta_model_b_minus_model_a': second_point[metric] - first_point[metric],
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'metric_direction': 'higher_is_better' if higher_is_better else 'lower_is_better',
            'bootstrap_probability_model_b_better': float(probability_b_better),
            'confidence_level': confidence_level,
            'n_resamples': n_resamples,
        })
    return pd.DataFrame(rows)


def cross_partition_duplicate_mask(train_frame, evaluation_frame, columns):
    """Flag exact evaluation rows whose selected values also occur in train."""
    missing = set(columns) - set(train_frame.columns) | set(columns) - set(evaluation_frame.columns)
    if missing:
        raise ValueError(f'Missing duplicate-comparison columns: {sorted(missing)}')
    if not columns:
        raise ValueError('Provide at least one duplicate-comparison column')
    train_hashes = pd.util.hash_pandas_object(train_frame[columns], index=False)
    evaluation_hashes = pd.util.hash_pandas_object(evaluation_frame[columns], index=False)
    return pd.Series(
        evaluation_hashes.isin(set(train_hashes)).to_numpy(),
        index=evaluation_frame.index,
        name='exact_duplicate_in_train',
    )
