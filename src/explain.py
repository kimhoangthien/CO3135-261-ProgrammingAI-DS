"""Global and representative local SHAP explanations for shared pipelines."""
import numpy as np
import pandas as pd

from .evaluate import PREDICTION_COLUMNS


def _take_rows(values, positions):
    if hasattr(values, 'iloc'):
        return values.iloc[positions]
    return np.asarray(values)[positions]


def _unwrap_estimator(pipeline):
    if not hasattr(pipeline, 'named_steps'):
        raise ValueError('Expected the shared fitted preprocessing/model pipeline')
    if not {'preprocessing', 'model'} <= set(pipeline.named_steps):
        raise ValueError('Pipeline must contain preprocessing and model steps')
    estimator = pipeline.named_steps['model']
    # The SMOTE strategy nests an imbalanced-learn pipeline as the model step.
    if hasattr(estimator, 'named_steps') and 'model' in estimator.named_steps:
        estimator = estimator.named_steps['model']
    return estimator


def _positive_class_explanation(raw, feature_names):
    import shap

    values = np.asarray(raw.values)
    base_values = np.asarray(raw.base_values)
    if values.ndim == 3:
        if values.shape[-1] != 2:
            raise ValueError('Expected binary SHAP outputs with two classes')
        values = values[..., 1]
        if base_values.ndim and base_values.shape[-1] == 2:
            base_values = base_values[..., 1]
    if values.ndim != 2:
        raise ValueError('Expected one SHAP value per observation and feature')
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=np.asarray(raw.data),
        feature_names=list(feature_names),
        output_names='SeriousDlqin2yrs=1',
    )


def explain_pipeline(pipeline, X_background, X_explain, background_size=200,
                     seed=42):
    """Explain positive-class scores after the fitted shared preprocessing.

    The background is sampled only from the caller-provided training rows. The
    caller controls which validation observations are explained; this function
    does not access the held-out test partition.
    """
    import shap

    if not isinstance(background_size, int) or background_size < 1:
        raise ValueError('background_size must be a positive integer')
    if len(X_background) == 0 or len(X_explain) == 0:
        raise ValueError('Background and explanation rows must be nonempty')
    classes = getattr(pipeline, 'classes_', None)
    if classes is None or list(classes) != [0, 1]:
        raise ValueError('Expected a fitted binary pipeline with classes [0, 1]')
    preprocessor = pipeline.named_steps.get('preprocessing')
    if preprocessor is None or not hasattr(preprocessor, 'feature_names_in_'):
        raise ValueError('Expected a fitted shared preprocessing step')
    estimator = _unwrap_estimator(pipeline)
    rng = np.random.default_rng(seed)
    size = min(background_size, len(X_background))
    positions = np.sort(rng.choice(len(X_background), size=size, replace=False))
    background = preprocessor.transform(_take_rows(X_background, positions))
    explained = preprocessor.transform(X_explain)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(estimator, background, feature_names=feature_names)
    raw = explainer(explained)
    return _positive_class_explanation(raw, feature_names)


def global_shap_importance(explanation):
    """Aggregate mean absolute SHAP values into a descending feature table."""
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite two-dimensional SHAP values')
    names = list(explanation.feature_names)
    if len(names) != values.shape[1]:
        raise ValueError('Feature names do not match SHAP columns')
    result = pd.DataFrame({
        'feature': names,
        'mean_absolute_shap': np.abs(values).mean(axis=0),
        'mean_shap': values.mean(axis=0),
    })
    return result.sort_values('mean_absolute_shap', ascending=False,
                              ignore_index=True)


def local_shap_table(explanation, row=0, top_n=10):
    """Return the strongest signed feature contributions for one observation."""
    values = np.asarray(explanation.values, dtype=float)
    data = np.asarray(explanation.data)
    if values.ndim != 2 or not 0 <= row < len(values):
        raise ValueError('row is outside the explanation')
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError('top_n must be a positive integer')
    result = pd.DataFrame({
        'feature': list(explanation.feature_names),
        'transformed_feature_value': data[row],
        'shap_value': values[row],
        'absolute_shap': np.abs(values[row]),
    })
    return result.sort_values('absolute_shap', ascending=False,
                              ignore_index=True).head(top_n)


def select_representative_cases(predictions, threshold=0.5, max_cases=3):
    """Choose high-risk, false-positive and false-negative validation examples."""
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError('Prediction schema differs from the shared contract')
    if not predictions['split'].eq('validation').all():
        raise ValueError('Representative cases must come from validation only')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    if not isinstance(max_cases, int) or not 1 <= max_cases <= 3:
        raise ValueError('max_cases must be between 1 and 3')
    frame = predictions.copy()
    frame['predicted_label'] = (frame['y_pred_proba'] >= threshold).astype(int)
    candidates = [
        ('highest_risk_positive', frame[frame['y_true'].eq(1)]
         .sort_values('y_pred_proba', ascending=False)),
        ('false_positive', frame[frame['y_true'].eq(0) & frame['predicted_label'].eq(1)]
         .sort_values('y_pred_proba', ascending=False)),
        ('false_negative', frame[frame['y_true'].eq(1) & frame['predicted_label'].eq(0)]
         .sort_values('y_pred_proba', ascending=False)),
    ]
    selected = []
    used = set()
    for case_type, candidates_frame in candidates:
        available = candidates_frame[~candidates_frame['sample_id'].isin(used)]
        if available.empty:
            continue
        row = available.iloc[0].to_dict()
        row['case_type'] = case_type
        selected.append(row)
        used.add(row['sample_id'])
        if len(selected) == max_cases:
            break
    if len(selected) < max_cases:
        remaining = frame[~frame['sample_id'].isin(used)].copy()
        remaining['distance_to_threshold'] = (
            remaining['y_pred_proba'] - threshold).abs()
        for _, fallback in remaining.sort_values('distance_to_threshold').iterrows():
            row = fallback.drop(labels='distance_to_threshold').to_dict()
            row['case_type'] = 'near_threshold'
            selected.append(row)
            if len(selected) == max_cases:
                break
    columns = ['case_type', *PREDICTION_COLUMNS, 'predicted_label']
    return pd.DataFrame(selected, columns=columns)
