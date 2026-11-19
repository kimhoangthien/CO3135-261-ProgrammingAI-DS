# Project audit recheck — 2026-11-19

## Gate decision

**Not yet approved for independent final test reporting.** The EDA notebook analyzes all labeled rows, including those subsequently assigned to the held-out partition. Whether that information influenced feature, model, or other development decisions has not been established. The team should agree on the intended claim and how to handle that exposure before freezing a protocol. The existing test partition has not been used for model performance evaluation in this recheck. No final results were exported and no models were retrained.

## Repository and scope

`dev` was at `7eacf41`, equal to the locally recorded `origin/dev`; no fetch was performed. The recorded data, baseline, XGBoost and evaluation feature tips are ancestors of `dev`; the cost feature tip equals `dev`. `main` remains at the foundation commit. Generated datasets, fitted models and results are ignored by Git and must be transferred separately. Existing uncommitted changes to `pyproject.toml` and `tests/test_evaluation.py` are the earlier test fix. README and team task status text is outdated.

The implementation and artifacts for all five members, notebooks 01–05, and the demo are present. This recheck covers code review, the safe test suite, and the validation-only artifact checks below. It is not a complete independent reproduction of all training, SHAP computations, figures, or EDA statistics. Historical assertions that test performance was never examined are based on project metadata, not independently provable from these files.

## Findings and recommended actions

| Severity | Module and evidence | Impact and recommendation | Retraining |
| --- | --- | --- | --- |
| Major | `notebooks/01_eda.ipynb`, cells 1–9 load `df = load_data()` and analyze all rows, including target-based distributions and correlations. The shared split is created from this same labeled file. | The eventual holdout has been exposed through aggregate descriptive analysis. This does not prove model performance was inspected or that results are invalid, but it limits a claim of a completely untouched holdout. Establish how the EDA influenced decisions and predeclare the reporting protocol. If an unexposed independent estimate is required, agree a fresh external evaluation or a revised design before use. | Not necessarily for a transparent report; a revised split-and-training design would require retraining. |
| Major, operational | `src/models/xgboost.py`, `src/models/logistic.py`, and `src/models/random_forest.py` use search `n_jobs=-1`; XGBoost and Random Forest estimators themselves use one worker. | Production tuning can launch multiple search workers despite single-worker estimators. Set and verify an explicit single-worker configuration before any rerun, especially given the reported native crash. The test invocation below limits joblib and native thread pools. | No for existing artifact inspection. |
| Minor, scientific | `src/split.py` checks unique positional IDs. The validation-only audit independently found 116 validation rows matching training rows on the ten features and target, including 3 positives. `results/tables/04_evaluation/duplicate_sensitivity.csv` contains a sensitivity analysis. | These are duplicate records, not overlapping sample IDs. They may make validation easier for repeated profiles; identical attributes do not prove the records belong to the same person. Retain and disclose the split and sensitivity analysis rather than silently deduplicating. | No for reporting the existing analysis; a group-aware redesign requires retraining. |
| Minor, scientific | `scripts/run_evaluation_explainability.py` selects calibration by validation Brier score; `scripts/run_cost_analysis.py` then optimizes thresholds on the same validation rows. | Results are development-stage and selection-optimistic. Freeze model, calibration, thresholds, costs, metrics and reporting before any approved evaluation. Bootstrap intervals condition on the selected models and fixed observed class proportions; they do not correct selection bias. | No. |
| Minor, scientific | `src/models/xgboost.py` computes training-wide class weights once before CV. SMOTE interpolates imputed values including missingness indicators. | Stratified folds make class ratios similar but the weight is not fold-specific; synthetic indicators can be fractional. Document both limitations. | No to document; potentially yes to change the experiment. |
| Minor, provenance | `results/tables/05_cost_analysis/provenance.json` records a dirty tree at generation and original/current hashes after folder migration. All ten currently referenced input hashes match their recorded current values; 17 prediction files reproduce from saved models. | These checks strongly support the current validation outputs, but do not prove that every historical change was path-only. Preserve the original provenance and retain this limitation. | Not indicated by current checks. |
| Presentation | `notebooks/02_baselines.ipynb` labels classes as default/no default, although `SeriousDlqin2yrs` represents serious delinquency. `notebooks/demo.ipynb` reads the absolute `figures` path stored in provenance. | Correct terminology in presentation materials; document the demo's machine-specific path or make it portable in a separately reviewed change. | No. |

The stored SHAP outputs explain the uncalibrated XGBoost pipeline, not the calibrated ensemble. Explainability is descriptive, not causal. Cost assumptions are hypothetical; all ratios are `C_FN:C_FP` and total cost is `FP*C_FP + FN*C_FN`.

## Validation and reproducibility checks

* `uv sync --locked`: passed after elevation for access to uv's cache; Python 3.12.7, uv 0.13.8.
* Safe suite, run outside the sandbox after prior Windows temp-directory permission errors:

  ```powershell
  $env:LOKY_MAX_CPU_COUNT='1'
  $env:OMP_NUM_THREADS='1'
  $env:OPENBLAS_NUM_THREADS='1'
  $env:MKL_NUM_THREADS='1'
  .\.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_real_data.py
  ```

  **108 passed in 30.99 seconds.** Known narrowly scoped dependency deprecations are filtered in `pyproject.toml`; no warnings were reported. `test_real_data.py` was excluded because it inspects held-out labels. Thus this is not a full-suite pass. Synthetic integration, XGBoost, calibration, SHAP and cost tests are included.
* `artifacts/audit_validation_recheck_20261119.py`, run with native thread limits, selected only train and validation CSV rows and did not evaluate test rows. The recorded IDs cover 150,000 positions uniquely: 105,000 train, 22,500 validation, 22,500 held-out. The four artifact split-manifest copies match the shared manifest byte for byte. The audit did not recompute the full-data fingerprint because doing so would inspect held-out values.
* All **17** validation prediction files had the expected sample IDs, labels, split and names, and reproduced from their corresponding 14 saved pipelines; largest absolute probability difference was about `1.2e-16`. This verifies inference, not independent reproduction of fitting.
* Cost comparison files were checked for six models and all four ratios: saved totals obey the cost formula and validation minima match a fresh sweep. The 116 duplicate overlap and three positive labels were independently reproduced on training/validation rows.
* The demo notebook executed in memory without saving or training: six code cells, 3.99 seconds. Jupyter emitted the Windows Proactor/ZeroMQ selector-thread runtime warning; execution succeeded. This checks availability on this machine, not presentation display quality on another machine.

## Implementation notes and remaining work

Member 1's data loader validates target and numeric feature schema, preserves rows, and the common preprocessing uses training-fitted median imputation with optional scaling and indicators. Member 2 tunes complete Logistic Regression and Random Forest pipelines using training CV and average precision. Member 3 uses stratified training CV, a training-derived negative/positive class weight and training-fold SMOTE. Member 4 computes ROC-AUC, average precision, classification metrics, Brier score and calibration on development data; Member 5 implements the exact cost sweep, deterministic tie handling and boundary decisions. These checks do not establish independent statistical or practical significance.

Before the audit can pass, resolve the holdout exposure and reporting protocol with the team, review the EDA's influence on decisions, complete any desired independent review of historical provenance and SHAP/figure interpretation, and freeze the selected comparison set and all evaluation decisions. Do not run a held-out evaluation without explicit user approval. If the gate passes, validation-only tables and figures can be assembled from the verified artifacts without retraining, labeled as development results and with the limitations above.