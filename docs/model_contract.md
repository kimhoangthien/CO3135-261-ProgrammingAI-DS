# Shared model contract

All future models expose `fit(X_train, y_train)` and `predict_proba(X)[:, 1]`.
Binary classes must be ordered `[0, 1]`; column 1 is serious delinquency probability.
Use the shared split manifest, the ten named features in `src.data.FEATURES`, and
positional `sample_id` (zero-based row position after index-column removal).
Keep row order fixed. The manifest fingerprint detects changed data or order.
Retain duplicate observations; document possible duplicate leakage before approving
any group-aware alternative to the required stratified baseline.

Fit the entire preprocessing plus estimator pipeline on training rows only.
Select hyperparameters, calibration and thresholds using training/validation only;
reserve test rows for the final locked evaluation. Never resample holdouts.
Use class weights when supported; any future SMOTE implementation belongs inside
a training-only imbalanced-learn pipeline, after splitting.

Save the complete fitted pipeline with `joblib.dump(pipeline, path)` and reload with
`joblib.load(path)` in the same pinned environment. Load only trusted artifacts.
Save alongside it the full YAML experiment configuration, estimator parameters,
random seeds (default 42), feature order, dependency freeze, dataset fingerprint,
Git revision when available, and split manifest. Verify reloaded probabilities
match the originals before accepting an artifact.

Export one row per observation with columns in this exact order:
`sample_id`, `y_true`, `y_pred_proba`, `model_name`, `split`.
Use integer sample IDs, binary integer true labels, finite probabilities in [0,1],
a stable model name, and split values `train`, `validation`, or `test`.
Do not include the target or sample ID in model features.
