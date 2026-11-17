"""Hypothetical validation cost analysis; never realized financial loss."""
import numpy as np
import pandas as pd
from .evaluate import PREDICTION_COLUMNS, validate_binary_probabilities


def validate_costs(c_fn, c_fp):
    costs = np.asarray([c_fn, c_fp], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any() or not costs.any():
        raise ValueError("Costs must be finite, nonnegative, and not both zero")
    return float(costs[0]), float(costs[1])


def threshold_cost_curve(y_true, probabilities, c_fn=1, c_fp=1):
    """Exact grouped-tie sweep. Positive iff p >= threshold.

    Zero is all-positive; nextafter(1,+inf) is all-negative even for p=1.
    Rows are descending by threshold, so exact cost ties prefer fewer positives.
    """
    y, p = validate_binary_probabilities(y_true, probabilities)
    c_fn, c_fp = validate_costs(c_fn, c_fp)
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(y)-1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, ends+1-np.cumsum(labels)[ends]]
    thresholds = np.r_[np.nextafter(1., np.inf), scores[ends]]
    if thresholds[-1] != 0:
        thresholds = np.r_[thresholds, 0.]
        tp, fp = np.r_[tp, tp[-1]], np.r_[fp, fp[-1]]
    fn, tn = y.sum()-tp, len(y)-y.sum()-fp
    total = c_fn*fn+c_fp*fp
    return pd.DataFrame(dict(threshold=thresholds, tn=tn, fp=fp, fn=fn, tp=tp,
                             total_cost=total, cost_per_observation=total/len(y)))


def optimal_threshold(y_true, probabilities, c_fn=1, c_fp=1):
    """Minimum empirical cost; exact ties choose the highest threshold."""
    curve = threshold_cost_curve(y_true, probabilities, c_fn, c_fp)
    return curve.loc[curve.total_cost.idxmin()].to_dict()


def validate_prediction_inputs(frames, validation_ids, validation_labels):
    """Require shared validation membership and labels; align by sample ID."""
    ids, labels = np.asarray(validation_ids), np.asarray(validation_labels)
    if ids.ndim != 1 or labels.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid validation reference")
    reference = pd.DataFrame(dict(sample_id=ids, y_true=labels)).sort_values("sample_id").reset_index(drop=True)
    if not frames:
        raise ValueError("Provide prediction frames")
    aligned = {}
    for name, frame in frames.items():
        if list(frame) != PREDICTION_COLUMNS:
            raise ValueError(f"{name}: prediction schema differs")
        if (not pd.api.types.is_integer_dtype(frame.sample_id)
            or frame.sample_id.duplicated().any() or (frame.sample_id < 0).any()
            or not frame.split.eq("validation").all()
            or not frame.model_name.eq(name).all()):
            raise ValueError(f"{name}: invalid IDs, model name or split")
        ordered = frame.sort_values("sample_id").reset_index(drop=True)
        if not ordered[["sample_id", "y_true"]].equals(reference):
            raise ValueError(f"{name}: validation IDs or labels differ")
        validate_binary_probabilities(ordered.y_true, ordered.y_pred_proba)
        aligned[name] = ordered
    return aligned
