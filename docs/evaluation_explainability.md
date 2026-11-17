# Evaluation and explainability — Member 4

## Scope and status

Member 4 owns consistent model evaluation, probability calibration and global/local
SHAP explanations. Development uses the shared training and validation partitions;
the held-out test partition remains locked. Threshold optimization and monetary
cost assumptions belong to Member 5.

The official 150,000-row Kaggle training CSV and the shared 70/15/15 split are
available locally. A full real-data run completed on 2026-11-17 using the trusted
Member 2/3 handoff pipelines configured in `configs/evaluation.yaml`. All model
and calibration choices below use validation only; the held-out test remains
unread by this workflow.

Isotonic calibration was selected for Logistic Regression, Random Forest and
standard XGBoost by validation Brier score with log loss as the tie-break. The
selected XGBoost candidate had ROC-AUC 0.8703, average precision 0.4030, Brier
score 0.0485 and log loss 0.1753. Random Forest was close: ROC-AUC 0.8690,
average precision 0.3994, Brier 0.0487 and log loss 0.1760. Their paired 95%
bootstrap intervals include zero for the ROC-AUC and average-precision deltas, so
the evidence does not support a strong claim that XGBoost dominates Random Forest.

The raw data contain 609 duplicate records. In the approved split, 116 validation
rows (0.516%) exactly match a training row on all features and target. Excluding
those rows for sensitivity analysis leaves the selected XGBoost ROC-AUC at 0.8702
and average precision at 0.4030. The shared split is not altered.

The full test suite completed with **85 passed**. This includes the official-data
contract test, synthetic end-to-end workflow, bootstrap/duplicate checks and SHAP
checks for Logistic Regression, Random Forest and XGBoost. SHAP 0.46 and its
plotting dependencies emit upstream deprecation warnings; no test failed.

## Implemented behavior

- `src/evaluate.py` preserves the existing metrics/prediction interfaces and adds
  log loss, generic split-aware prediction frames, and paired comparison of
  prediction files aligned by stable `sample_id`. It also provides class-stratified
  percentile bootstrap intervals, paired model deltas and exact cross-partition
  duplicate detection.
- `src/calibration.py` fits sigmoid and isotonic calibration with stratified folds
  from training data only. Reliability tables include counts, observed rates,
  expected calibration error and maximum calibration error. Calibration candidates
  are selected on validation Brier score, then log loss as a deterministic tie-break.
- `src/explain.py` explains fitted shared preprocessing/model pipelines. SHAP
  background rows are sampled only from training data; outputs are normalized to
  positive class `SeriousDlqin2yrs=1`. It exports global mean absolute importance
  and 2–3 representative validation cases (high-risk positive, false positive and
  false negative when available). SHAP targets the underlying uncalibrated ranking
  model; probability calibration is reported as a separate post-processing layer.
  Local feature values are explicitly labeled as transformed values.
- `scripts/run_evaluation_explainability.py` consumes the trusted tuned Logistic
  Regression, Random Forest and standard XGBoost artifacts. It saves calibrated
  pipelines, validation prediction files, calibration bins, comparison tables,
  bootstrap intervals, duplicate-sensitivity results, SHAP tables, reproducibility
  metadata and the unchanged split manifest under the existing Member 4 output
  directories.
- `notebooks/04_evaluation.ipynb` renders comparison and uncertainty tables,
  ROC/PR curves, confusion matrices, reliability diagrams, global SHAP importance
  and local signed-contribution plots without fitting models. Executing it saves
  the five report figures under `results/figures/04_evaluation/`.

ECE and MCE are descriptive bin-dependent summaries, not proper scoring rules.
Brier score and log loss are used for calibration selection. SHAP describes model
behavior and must not be presented as a causal explanation or reason for denying
credit.

## Required handoff and execution

Expected trusted artifacts:

- `artifacts/02_baselines/logistic_tuned.joblib`
- `artifacts/02_baselines/random_forest_tuned.joblib`
- `artifacts/03_xgboost/xgboost_standard_tuned.joblib`
- official `data/raw/cs-training.csv` and fingerprint-matching
  `data/processed/splits.json`

Run from the repository root:

```powershell
uv run python scripts/run_evaluation_explainability.py
uv run python -m nbconvert --to notebook --execute notebooks/04_evaluation.ipynb `
  --ExecutePreprocessor.kernel_name=credit-risk `
  --ExecutePreprocessor.timeout=600 `
  --output 04_evaluation.executed.ipynb --output-dir artifacts
uv run pytest -q
```

Use `--skip-shap` only for a calibration smoke run. Use `--models` to limit a
smoke run; a reportable run must include all agreed comparison candidates.

## Handoff

Member 5 should consume the selected validation prediction files and state all
cost ratios as `C_FN:C_FP`. Thresholds must be chosen on validation only. Member 1
should perform the single final test evaluation only after model, calibration and
cost-threshold decisions are approved and frozen.
