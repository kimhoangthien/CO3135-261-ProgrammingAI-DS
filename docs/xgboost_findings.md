# XGBoost and class-imbalance findings (Member 3)

Experiments E2 (imbalance handling) and E3 (tuning), plus the XGBoost part of E1.
All figures below are **validation-only** results from one full run; the test partition
was never read. They are development evidence, not final performance claims.

## Run provenance

| Item | Value |
|---|---|
| Code | commit `912ce05`, no uncommitted changes at run time |
| Data | official `cs-training.csv`, fingerprint `749ec0258bc5cf33…` (matches `eda_findings.md`) |
| Splits | shared manifest, seed 42, stratified 70/15/15: 105,000 train (7,018 positives) / 22,500 validation (1,504 positives) |
| Tuning | `RandomizedSearchCV`, 20 candidates per strategy, stratified 3-fold CV on train, scoring average precision |
| Libraries | Python 3.12.7, xgboost 2.1.4, imbalanced-learn 0.12.4, scikit-learn 1.5.2 |
| Runtime | about 3 minutes in total on a laptop |

The full configuration, estimator parameters, dataset fingerprint, Git revision and
dependency freeze are saved in `artifacts/member3/configuration.yaml` and
`artifacts/member3/dependencies.txt`.

## Method

- **Pipeline.** The shared `build_model_pipeline` (train-fitted median imputation with
  missing-value indicators, no scaling) followed by `XGBClassifier`
  (`binary:logistic`, `hist`, `random_state=42`). Step names `preprocessing` / `model`
  follow the shared contract, so `verify_preprocessing` applies unchanged.
- **Strategies.**
  - `standard`: no imbalance handling.
  - `weighted`: `scale_pos_weight` = negatives / positives on the training partition
    (13.96). The shared `balanced=True` option is not used because XGBoost has no
    `class_weight`.
  - `smote`: `SMOTE(random_state=42)` placed with XGBoost in a nested imbalanced-learn
    pipeline used as the final estimator. Resampling therefore happens only inside `fit`
    (each training fold during CV, the full training partition at refit) and never at
    prediction time. An imbalanced-learn pipeline cannot contain the shared sklearn
    preprocessing pipeline as an intermediate step, hence the nesting.
- **Search space** (`configs/xgboost.yaml`): `n_estimators` {200, 400, 600},
  `max_depth` {3–6}, `learning_rate` {0.01–0.2}, `subsample` and `colsample_bytree`
  {0.6–1.0}, `min_child_weight` {1, 3, 5, 10}, `reg_lambda` {0.1, 1, 5, 10}. The same
  seed draws the **same 20 candidates for every strategy**, so E2 compares strategies
  under an identical tuning budget.
- **Reporting threshold** fixed at 0.5 for precision / recall / F1, as in the baselines.
  Threshold selection is Member 5's cost analysis.

## Results (validation, 22,500 rows, observed delinquency rate 0.0668)

| Model | Experiment | CV AP | ROC-AUC | AP | Brier | Mean predicted prob. | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|---|---|---|---|
| `xgboost_default` | E3 | – | 0.8612 | 0.3781 | 0.0502 | 0.0666 | 0.556 | 0.201 | 0.296 |
| `xgboost_standard_tuned` | E2, E3 | 0.4008 | 0.8701 | **0.4042** | **0.0485** | 0.0668 | 0.591 | 0.182 | 0.278 |
| `xgboost_weighted_tuned` | E2 | 0.3973 | 0.8704 | **0.4042** | 0.1392 | 0.3203 | 0.217 | 0.783 | 0.340 |
| `xgboost_smote_tuned` | E2 | 0.3849 | 0.8658 | 0.3889 | 0.0499 | 0.0863 | 0.574 | 0.228 | 0.326 |

AP is average precision (step-wise PR-AUC), as defined in `src/evaluate.py`.
Confusion matrices at 0.5 (TN / FP / FN / TP): default 20,754 / 242 / 1,201 / 303;
standard tuned 20,806 / 190 / 1,230 / 274; weighted tuned 16,756 / 4,240 / 327 / 1,177;
SMOTE tuned 20,741 / 255 / 1,161 / 343.

Selected configuration for `standard` and `weighted` (identical): `n_estimators=600`,
`max_depth=5`, `learning_rate=0.01`, `subsample=0.6`, `colsample_bytree=0.7`,
`min_child_weight=3`, `reg_lambda=5`. For `smote`: `n_estimators=400`, `max_depth=5`,
`learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.6`, `min_child_weight=1`,
`reg_lambda=5`.

### E3: tuning is worthwhile, moderately

- Tuning raises validation AP from 0.378 to 0.404 (+0.026) and ROC-AUC from 0.861 to
  0.870, and slightly improves the Brier score (0.0502 to 0.0485).
- Configuration choice matters more than the average gain suggests: mean CV AP across the
  20 sampled configurations spans 0.344–0.401 (`standard`), 0.335–0.397 (`weighted`) and
  0.342–0.385 (`smote`). A poor configuration loses more than tuning gains over defaults.
- The CV fold standard deviation of the selected `standard` configuration is about 0.004.

### E2: imbalance handling does not improve ranking

- **`scale_pos_weight`** selects the same hyperparameters and reaches the same ROC-AUC
  and AP as standard training. It does not rank borrowers better. It shifts every
  probability upward (mean 0.320 against an observed rate of 0.067), roughly triples the
  Brier score, and its reliability curve lies far below the diagonal. Its high recall at
  0.5 (0.78) is a threshold effect. The two models rank borrowers almost identically
  (Spearman correlation 0.997 between their validation scores). Flagging the same number
  of borrowers (5,417) with the standard model, at a threshold of about 0.070, gives
  precision 0.218 / recall 0.785, against 0.217 / 0.783 for the weighted model at 0.5.
  This is a descriptive check, not a threshold selection; choosing the threshold from
  costs on validation is the cost-sensitive approach (Elkan, 2001) owned by Member 5.
- **SMOTE** lowers AP (0.389), takes about 2.7x longer to tune (94 s vs 35 s), and mildly
  overestimates risk (mean predicted probability 0.086).
- **Conclusion.** On this dataset, class imbalance is better addressed at decision time
  (threshold choice on validation) than during training. Training-time reweighting mainly
  changes the probability scale.

### E1 (XGBoost part): tied with Random Forest

| Model | Source | ROC-AUC | AP | Brier |
|---|---|---|---|---|
| Logistic Regression (balanced baseline) | Member 1, `eda_findings.md` | 0.8021 | 0.3136 | 0.1794 |
| Random Forest (tuned) | Member 2, `notebooks/02_baselines.ipynb` | 0.8694 | 0.4013 | 0.0487 |
| XGBoost (tuned, standard) | this run | 0.8701 | 0.4042 | 0.0485 |

The XGBoost margin over Random Forest (+0.003 AP, +0.001 ROC-AUC) is smaller than the CV
fold standard deviation, so **no superiority claim is made**: the two tree ensembles are
effectively tied on validation, and both clearly outperform Logistic Regression. The
Random Forest figures are copied from Member 2's notebook; a direct comparison from
prediction files is in `notebooks/03_xgboost.ipynb` once Member 2's validation predictions
are available locally.

### Probability reliability (preliminary)

A decile reliability diagram (`results/figures/member3/reliability.png`) shows the
`standard` model close to the diagonal, `smote` slightly above it in predicted risk, and
`weighted` strongly overestimating. Formal calibration (E4) is Member 4's scope; nothing
is calibrated here.

### Feature importance (gain, not causal)

Share of total split gain in `xgboost_standard_tuned`: `NumberOfTimes90DaysLate` 0.311,
`RevolvingUtilizationOfUnsecuredLines` 0.294, `NumberOfTime30-59DaysPastDueNotWorse`
0.133, `NumberOfTime60-89DaysPastDueNotWorse` 0.108, then `age` 0.039, `DebtRatio` 0.035,
`NumberOfOpenCreditLinesAndLoans` 0.027, `MonthlyIncome` 0.026,
`NumberRealEstateLoansOrLines` 0.020, `NumberOfDependents` 0.005, and the two
missing-value indicators below 0.003. Past-delinquency counts include the 96/98 values
flagged in `eda_findings.md`, which trees can isolate; this should be cross-checked with
SHAP. Gain importance describes the model, not causes of delinquency.

## Handoff

**Member 4 (evaluation, calibration, SHAP)**

- Recommended XGBoost candidate: `xgboost_standard_tuned` (best AP and Brier, mean
  probability matching the observed rate).
- `xgboost_weighted_tuned` is a clear example of distorted probabilities for E4.
- Pipelines: `artifacts/member3/<model_name>.joblib` (full preprocessing + model, reload
  verified). For the SMOTE model, the XGBoost estimator is at
  `pipeline.named_steps["model"].named_steps["model"]`; the other models expose it at
  `pipeline.named_steps["model"]`.
- Predictions: `results/tables/member3/<model_name>_validation_predictions.csv`, columns
  `sample_id, y_true, y_pred_proba, model_name, split`, rows in shared manifest order.
- Also available: `metrics.json`, `summary.csv`, `cv_results_<strategy>.csv`.
- Generated files are ignored by Git; transfer them through the team's private channel
  or regenerate them (see below).

**Member 5 (cost analysis)**

- Use calibrated or well-scaled probabilities for threshold sweeps. `weighted` probabilities
  are not comparable with the others at the same threshold.

## Reproduce

From the repository root, with the official CSV and the shared `data/processed/splits.json`:

```powershell
uv run python scripts/run_xgboost_experiments.py
uv run python -m nbconvert --to notebook --execute notebooks/03_xgboost.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output 03_xgboost.executed.ipynb --output-dir artifacts
uv run pytest tests/test_xgboost.py
```

`--n-iter 2` gives a quick smoke run; its outputs are flagged `smoke_run: true` and the
notebook refuses to analyse them. Results are deterministic for a fixed seed and library
versions.

## Tests

`tests/test_xgboost.py` (20 tests, synthetic fixtures only) covers the model contract for
each strategy, `scale_pos_weight` computation and validation, SMOTE resampling only during
training (call spy) and never for other strategies, joblib reload equality, seed
reproducibility, tuning parameter prefixes, and consistency between
`configs/xgboost.yaml` and the code. Re-injecting three known bugs made the suite fail
each time.

## Limitations

- Single seed (42) and a single validation partition; no repeated-split variance estimate.
- Selecting the best of several models on the same validation partition makes its score
  slightly optimistic. Final claims require the locked test evaluation coordinated by
  Member 1.
- `scale_pos_weight` is computed once on the full training partition and reused inside
  each CV fold; stratified folds make the per-fold ratio nearly identical.
- SMOTE interpolates in the imputed feature space, including the binary missing-value
  indicators, which can take fractional values in synthetic rows.
- Precision / recall / F1 at 0.5 are not comparable across strategies with different
  probability scales.
- Duplicate observations (609, see `eda_findings.md`) may cross partitions; they are
  retained, as for all models.
- The target is serious delinquency within two years, not default or financial loss.
  Nothing here supports causal or monetary conclusions.
