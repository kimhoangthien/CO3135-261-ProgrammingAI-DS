"""Minimal validation metrics for the initial baselines; Member 4 owns extensions."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

PREDICTION_COLUMNS = ['sample_id', 'y_true', 'y_pred_proba', 'model_name', 'split']


def evaluate_probabilities(y_true, probabilities, threshold=0.5):
    """PR-AUC here means average precision (step-weighted area, not trapezoidal)."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('Expected aligned binary labels with both classes and finite probabilities in [0,1]')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    predicted = p >= threshold
    return {
        'probability_metrics': {
            'roc_auc': float(roc_auc_score(y, p)),
            'pr_auc_average_precision': float(average_precision_score(y, p)),
            'brier_score': float(brier_score_loss(y, p)),
        },
        'threshold_metrics': {
            'threshold': float(threshold),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        },
    }


def validation_predictions(sample_ids, y_true, probabilities, model_name):
    evaluate_probabilities(y_true, probabilities)
    ids = np.asarray(sample_ids)
    if (len(ids) != len(y_true) or ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids)) != len(ids) or np.any(ids < 0)):
        raise ValueError('Sample IDs must be unique, nonnegative integer row positions')
    return pd.DataFrame(dict(sample_id=ids, y_true=np.asarray(y_true, dtype=int),
                             y_pred_proba=probabilities, model_name=model_name,
                             split='validation'), columns=PREDICTION_COLUMNS)
