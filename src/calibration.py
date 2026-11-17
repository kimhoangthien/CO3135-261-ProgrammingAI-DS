"""Leakage-safe calibration fitting and probability reliability summaries."""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from .evaluate import evaluate_probabilities, validate_binary_probabilities


CALIBRATION_METHODS = ('sigmoid', 'isotonic')


def build_calibrated_classifier(estimator, method='sigmoid', cv=3, seed=42):
    """Clone and calibrate an estimator using folds from training data only.

    The returned classifier is unfitted. Calling ``fit(X_train, y_train)`` fits
    preprocessing, the estimator and its calibrators inside stratified training
    folds. Validation and test observations must never be passed to ``fit``.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f'Method must be one of {CALIBRATION_METHODS}')
    if not isinstance(cv, int) or cv < 2:
        raise ValueError('cv must be an integer of at least 2')
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        estimator=clone(estimator), method=method, cv=folds, ensemble=True,
    )


def calibration_table(y_true, probabilities, n_bins=10, strategy='quantile'):
    """Return nonempty reliability bins with gaps and observation counts."""
    y, p = validate_binary_probabilities(y_true, probabilities)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError('n_bins must be an integer of at least 2')
    if strategy not in {'uniform', 'quantile'}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    if strategy == 'uniform':
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) == 1:
            edges = np.array([edges[0], edges[0]])
    bin_ids = np.searchsorted(edges[1:-1], p, side='right')
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        mean_probability = float(p[mask].mean())
        observed_rate = float(y[mask].mean())
        rows.append({
            'bin_id': bin_id,
            'lower_bound': float(edges[bin_id]),
            'upper_bound': float(edges[bin_id + 1]),
            'count': int(mask.sum()),
            'mean_predicted_probability': mean_probability,
            'observed_positive_rate': observed_rate,
            'absolute_gap': abs(mean_probability - observed_rate),
        })
    return pd.DataFrame(rows, columns=[
        'bin_id', 'lower_bound', 'upper_bound', 'count',
        'mean_predicted_probability', 'observed_positive_rate', 'absolute_gap',
    ])


def evaluate_calibration(y_true, probabilities, n_bins=10, strategy='quantile'):
    """Return proper scoring rules plus ECE/MCE descriptive reliability gaps."""
    metrics = deepcopy(evaluate_probabilities(y_true, probabilities))
    bins = calibration_table(y_true, probabilities, n_bins, strategy)
    weights = bins['count'].to_numpy(dtype=float) / bins['count'].sum()
    gaps = bins['absolute_gap'].to_numpy(dtype=float)
    metrics['calibration_metrics'] = {
        'n_bins_requested': n_bins,
        'n_bins_observed': len(bins),
        'strategy': strategy,
        'expected_calibration_error': float(np.sum(weights * gaps)),
        'maximum_calibration_error': float(gaps.max()),
    }
    return metrics, bins


def select_calibration_candidate(metrics_by_name):
    """Select the smallest validation Brier score with deterministic tie-breaks."""
    if not metrics_by_name:
        raise ValueError('Provide at least one calibration candidate')
    scored = []
    for name, metrics in metrics_by_name.items():
        try:
            brier = float(metrics['probability_metrics']['brier_score'])
            logloss = float(metrics['probability_metrics']['log_loss'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'{name}: malformed probability metrics') from None
        if not np.isfinite([brier, logloss]).all():
            raise ValueError(f'{name}: calibration scores must be finite')
        scored.append((brier, logloss, name))
    return min(scored)[2]
