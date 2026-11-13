# Team assignments and ownership

Member 1 is the team leader. Member identifiers below are the team's shared labels;
branch names contain no personal identifiers. Read CONTRIBUTING.md before starting.

| Member | Assigned branch | Ownership |
|---|---|---|
| Member 1 | feature/data-pipeline | Shared data, minimal baselines, integration QA |
| Member 2 | feature/baseline-models | Logistic Regression tuning and Random Forest |
| Member 3 | feature/xgboost-imbalance | XGBoost and imbalance experiments |
| Member 4 | feature/evaluation-explainability | Evaluation, calibration and SHAP |
| Member 5 | feature/cost-analysis-demo | Hypothetical costs and notebook demonstration |

All five branches must start at the same validated dev commit, independently
created from dev. PRs target dev. Member 1 reviews integration; main stays stable.
Branch publication must be verified against origin before members start checkout.

## Member 1: Data Pipeline & Integration (leader)

Own src/data.py, src/split.py, src/preprocessing.py, src/train.py,
notebooks/01_eda.ipynb, shared data documentation, research overview/methodology,
model contract, integration tests, and initial baseline maintenance. Coordinate
changes to configs/default.yaml and configs/baselines.yaml. The foundation lives
on main; feature/data-pipeline is for further development and maintenance.

The shared data pipeline and initial Dummy prior/Logistic Regression baselines
have been executed on the local official dataset. EDA findings, split fingerprint,
validation-only metrics and artifacts are documented in `docs/eda_findings.md`.
The fixed test partition remains unused. Reuse the validated manifest; do not
replace it without team approval. Never treat fixture results as research results.

## Member 2: Baseline Models

Own src/models/logistic.py after coordinating the initial factory handoff,
src/models/random_forest.py, notebooks/02_baselines.ipynb and relevant model tests.
Reuse build_logistic_baseline instead of creating a second baseline implementation.
Tune Logistic Regression and implement/tune Random Forest; record configurations,
persist full fitted pipelines, export validation predictions/metrics, and prepare
initial comparison tables. Proposed RF files/notebook are future work, not present
experiments. Coordinate shared training/evaluation changes with Members 1 and 4.

## Member 3: XGBoost & Class Imbalance

Own future src/models/xgboost.py, notebooks/03_xgboost.ipynb and relevant tests.
Implement/tune XGBoost and compare class weighting and justified SMOTE strategies.
Use shared splits and preprocessing; apply SMOTE only inside training folds and
never to validation/test observations. Save configs, full pipelines, validation
predictions and metrics. Coordinate new dependencies with Member 1; no unreviewed
upgrades or independent split definitions.

## Member 4: Evaluation & Explainability

Own src/evaluate.py, src/calibration.py, src/explain.py, future
notebooks/04_evaluation.ipynb and relevant tests. Preserve the minimal metrics
interface used by src/train.py while adding ROC-AUC, PR-AUC, precision, recall,
F1, Brier score, confusion matrices, calibration curves and global/local SHAP.
Accept the shared prediction schema and trusted fitted pipeline artifacts.
Document whether PR-AUC is average precision or trapezoidal area; initial baselines
use average precision. Keep probability and threshold metrics separate. Do not
change model implementations without coordination. Calibration uses development
partitions/folds; never fit or select using the held-out test partition.

## Member 5: Cost Analysis & Demo

Own src/cost_analysis.py, future notebooks/05_cost_analysis.ipynb,
notebooks/demo.ipynb and relevant tests. Use clearly labeled hypothetical
false-positive/false-negative costs, choose thresholds on validation only, and
produce cost curves, sensitivity analysis, final demonstration and visual outputs.
Do not claim measured bank profit or realized monetary-loss optimization.
Never choose thresholds from the held-out test set.

## Shared boundaries and handoff order

1. Member 1 authenticates the official download, executes EDA, reviews data risks,
   and validates the unchanged 70/15/15 stratified seed-42 manifest and train-only
   preprocessing. No row deletion, clipping, feature changes or sentinel recoding
   without documented evidence and team review.
2. Member 1 runs the minimal baselines and verifies persisted pipeline reloads.
   Members 2 and 3 can develop on their assigned branches in parallel, but all
   comparable experiments must use that same manifest and feature order.
3. Members 2/3 hand full pipelines, configuration, random seeds, dependency versions,
   dataset fingerprint and validation predictions to Member 4.
4. Member 4 delivers consistent evaluation, calibration and explanation outputs.
5. Member 5 integrates cost scenarios and the notebook demonstration. Member 1
   reviews dev integration, locks model/threshold decisions, and coordinates one
   final held-out test evaluation after development is complete.

Positive class: SeriousDlqin2yrs=1, serious delinquency (90+ days past due or worse)
within two years, not legal default, bankruptcy or realized financial loss.
Predict probability with predict_proba(X)[:, 1] and class order [0,1].
Export columns exactly: sample_id, y_true, y_pred_proba, model_name, split.
Use zero-based positional raw-row sample IDs, never reset IDs after splitting.
The unchanged contract is in model_contract.md. Notify Member 1 before editing
another member's primary files; Member 1 coordinates shared-file integration.
