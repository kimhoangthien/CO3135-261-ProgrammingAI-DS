# Team assignments and ownership

Project: **Explainable and Cost-Sensitive Credit Risk Scoring Using Machine Learning**.

Member 1 is the team leader. Member identifiers below are the team's shared labels;
branch names contain no personal identifiers. Read [CONTRIBUTING.md](CONTRIBUTING.md) before starting.

| Member | Assigned branch | Ownership |
|---|---|---|
| Member 1 | feature/data-pipeline | Shared data, minimal baselines, integration QA |
| Member 2 | feature/baseline-models | Logistic Regression and Random Forest |
| Member 3 | feature/xgboost-imbalance | XGBoost and imbalance experiments |
| Member 4 | feature/evaluation-explainability | Evaluation, calibration and explainability |
| Member 5 | feature/cost-analysis-demo | Cost-sensitive analysis and demonstration |

## Mandatory Git branch structure

```text
main
└── dev
    ├── feature/data-pipeline
    ├── feature/baseline-models
    ├── feature/xgboost-imbalance
    ├── feature/evaluation-explainability
    ├── feature/cost-analysis-demo
```

`main` is the stable release branch; `dev` is the integration branch. All five
feature branches must be created **directly from the same validated `dev` commit**,
never from `main` or another feature. They are prepared centrally by Member 1.
Members check out their assigned existing branch, not create a new branch.
Every feature Pull Request targets **`dev`**; Member 1 reviews integration.

Fetch and inspect the remote branches, then check out your assigned branch:

```powershell
git fetch origin
git branch -r
# Example for Member 2; use your exact assigned name from the overview table.
git switch --track origin/feature/baseline-models
# If the assigned branch already exists locally, use this instead:
git switch feature/baseline-models
uv sync --locked
```

If a remote branch is missing, ask Member 1 to verify publication. Do not recreate
it from another starting point. See CONTRIBUTING.md for every member's checkout.

## Member 1: Data Pipeline & Integration (leader)

**Branch:** `feature/data-pipeline`

**Responsibilities**

- Integrate the official Kaggle Give Me Some Credit dataset; maintain downloading,
  authentication/permission handling, schema and target validation.
- Perform EDA of missing values, duplicates, outliers, suspicious values,
  correlations, class imbalance and feature-target relationships; export figures
  and summary tables and document actual findings.
- Maintain reproducible stratified 70/15/15 partitions with seed 42, persistent
  identifiers, fingerprint checks and non-overlapping split validation.
- Maintain shared preprocessing with training-only fitting, imputation, optional
  indicators, appropriate scaling and consistent feature order.
- Implement and maintain initial Dummy Classifier and Logistic Regression baselines
  and their shared training/persistence workflow.
- Maintain the model contract and prediction schema; coordinate repository
  integration, code review, tests, QA, research overview and methodology.

**Implementation files:** `src/data.py`, `src/split.py`, `src/preprocessing.py`,
`scripts/download_dataset.py`, `notebooks/01_eda.ipynb`, `docs/eda_findings.md`.
The actual initial baseline workflow is in **`src/train.py`** (Dummy construction,
fitting, export and persistence), **`src/models/logistic.py`** (Logistic Regression
factory), and **`src/evaluate.py`** (minimal metrics/export interface).
`configs/baselines.yaml` records baseline settings; `tests/test_baselines.py`
checks the shared workflow. Module ownership and coordinated handoffs are below.

**Deliverables:** validated official dataset, real-data EDA findings, reproducible
splits, preprocessing pipeline, initial baseline results, and shared documentation.
The foundation lives on `main`; this feature branch is for ongoing maintenance.

**Recorded status:** initial code provides `DummyClassifier(strategy="prior")` and
a scaled, balanced Logistic Regression factory, verified using unit fixtures.
Real-data EDA, fitting, metrics and artifacts remain blocked in the latest recorded
validation by Kaggle authentication. Resolve access and validate shared splits
before publishing experimental findings. Never substitute fixtures for real data.

## Member 2: Logistic Regression & Random Forest

**Branch:** `feature/baseline-models`

**Responsibilities**

- Extend and tune the existing Logistic Regression baseline; reuse
  `build_logistic_baseline` rather than duplicate it.
- Implement and optimize Random Forest using cross-validation on training data.
- Compare Logistic Regression and Random Forest on validation data using the
  shared evaluation interface, split manifest and feature definitions.
- Persist complete fitted model pipelines and export validation predictions,
  metrics and reproducible experiment configurations.

**Implementation files:** existing `src/models/logistic.py`; planned
`src/models/random_forest.py`, `notebooks/02_baselines.ipynb`, and
`tests/test_baseline_models.py`. Existing `tests/test_baselines.py` covers the
initial shared workflow; it is not renamed or replaced by the planned model tests.

**Deliverables:** tuned Logistic Regression, Random Forest, validation comparisons,
model artifacts, prediction files and configurations. Coordinate the existing
Logistic Regression factory handoff with Member 1 and evaluation changes with
Member 4. No test-set selection or extensive search outside the agreed scope.

## Member 3: XGBoost & Class Imbalance

**Branch:** `feature/xgboost-imbalance`

**Responsibilities**

- Implement and tune XGBoost using training cross-validation.
- Investigate class weighting and `scale_pos_weight`; compare imbalance strategies.
- Evaluate justified SMOTE experiments **inside training folds only**; never
  resample validation or test observations.
- Save full trained pipelines, configurations, validation metrics/predictions,
  and documented experimental findings.

**Planned implementation files:** `src/models/xgboost.py`,
`notebooks/03_xgboost.ipynb`, `tests/test_xgboost.py`.

**Deliverables:** optimized XGBoost, documented imbalance experiments, validation
metrics, fitted pipelines and prediction files. Reuse shared splits/preprocessing;
coordinate dependency changes with Member 1. No independent split definitions.

## Member 4: Evaluation, Calibration & Explainability

**Branch:** `feature/evaluation-explainability`

**Responsibilities**

- Extend reusable evaluation utilities while preserving the minimal interface
  consumed by `src/train.py`.
- Calculate ROC-AUC, PR-AUC, precision, recall, F1-score, Brier score and confusion
  matrices; generate ROC and precision-recall curves and comparison tables.
- Evaluate probability calibration and implement appropriate calibration methods
  and curves using development partitions/folds only.
- Produce global and local SHAP explanations and model-comparison visualizations.

**Implementation files:** existing `src/evaluate.py`; existing documentation-only
placeholders `src/calibration.py` and `src/explain.py`; planned
`notebooks/04_evaluation.ipynb` and `tests/test_evaluation.py`.

**Deliverables:** evaluation results, comparison tables, calibration analysis,
SHAP figures and **2-3 representative local explanations**. Accept the shared
prediction schema and trusted fitted pipelines. Initial baseline PR-AUC uses
average precision; explicitly distinguish it from trapezoidal PR area. Report
probability metrics separately from threshold metrics. Do not change another
member's model implementation without agreement, or fit/select using test data.

## Member 5: Cost-Sensitive Analysis & Demo

**Branch:** `feature/cost-analysis-demo`

**Responsibilities**

- Implement cost-sensitive classification analysis with clearly labeled
  hypothetical false-positive/false-negative assumptions.
- Compare ratios such as **1:1, 3:1, 5:1 and 10:1**. Label ratio orientation
  explicitly, for example `C_FN:C_FP`, and keep it consistent across comparisons.
- Select cost-minimizing thresholds from validation predictions; generate cost
  curves, threshold comparisons and sensitivity analyses across models.
- Build a reproducible Jupyter demonstration integrating predictions, evaluation
  figures, SHAP explanations and cost-analysis results.

**Implementation files:** existing documentation-only `src/cost_analysis.py`;
existing Phase 1 `notebooks/demo.ipynb` (shared-data demonstration, not the complete
final demo); planned `notebooks/05_cost_analysis.ipynb` and
`tests/test_cost_analysis.py`.

**Deliverables:** cost-sensitive results, threshold comparisons, sensitivity plots
and the complete demonstration notebook. All monetary assumptions are hypothetical;
do not present them as actual bank profits or realized financial losses. Never
choose thresholds using the held-out test partition.

## Shared technical standards

| Component | Standard |
|---|---|
| Package manager | `uv`; `pyproject.toml` and `uv.lock` are authoritative |
| Environment setup | `uv sync --locked`; preserve the pinned Python 3.12.7 environment |
| Dataset | Official Kaggle Give Me Some Credit; `data/raw/cs-training.csv` |
| Data splitting | Stratified 70/15/15, seed 42; shared fingerprint-validated manifest |
| Preprocessing | Fit only on training data; refit within each training CV fold |
| Model selection | Training cross-validation and validation data |
| Final evaluation | Untouched held-out test data, after development decisions are locked |
| Model persistence | Complete fitted preprocessing plus estimator pipeline; verify reload probabilities |
| Prediction schema | `sample_id, y_true, y_pred_proba, model_name, split` |
| Testing | `uv run pytest` |
| Integration | Feature Pull Requests into `dev` |
| Shared code changes | Coordinate with and obtain review from Member 1 |

`SeriousDlqin2yrs=1` means serious delinquency (90 days past due or worse) within
two years, not legal default, bankruptcy or realized financial loss. Models expose
`fit(X_train, y_train)` and `predict_proba(X)[:, 1]` with class order `[0, 1]`.
Use zero-based positional sample identifiers from the ordered cleaned raw dataframe;
never reset IDs after splitting or include IDs/target in features.

Persist configuration, seed, feature order, dataset fingerprint, split manifest,
Git revision and dependency versions alongside fitted pipelines. See the unchanged
[model contract](model_contract.md). No resampling of holdouts, silent row deletion,
clipping, feature changes or sentinel recoding without evidence and team review.
Keep datasets, credentials and generated artifacts out of Git.

## File ownership and coordination

Paths marked **planned** do not exist yet; **placeholder** means the file exists
but contains no implemented experiment. A library being installed does not establish
that a model or analysis has been implemented.

| Primary owner | Existing implementation files | Planned files / existing placeholders |
|---|---|---|
| Member 1 | `src/data.py`, `src/split.py`, `src/preprocessing.py`, `src/config.py`, `scripts/download_dataset.py`, `src/train.py`, `configs/default.yaml`, `configs/baselines.yaml`, `notebooks/01_eda.ipynb`, `docs/eda_findings.md`, `docs/model_contract.md`, `tests/test_data.py`, `tests/test_split.py`, `tests/test_preprocessing.py`, `tests/test_download_dataset.py`, `tests/test_baselines.py`, `tests/test_real_data.py` | No new file required for current ownership |
| Member 2 | `src/models/logistic.py` (initial factory maintained with Member 1 coordination) | **Planned:** `src/models/random_forest.py`, `notebooks/02_baselines.ipynb`, `tests/test_baseline_models.py` |
| Member 3 | No dedicated experiment implementation yet | **Planned:** `src/models/xgboost.py`, `notebooks/03_xgboost.ipynb`, `tests/test_xgboost.py` |
| Member 4 | `src/evaluate.py` (initial metrics used by Member 1's workflow) | **Placeholders:** `src/calibration.py`, `src/explain.py`; **planned:** `notebooks/04_evaluation.ipynb`, `tests/test_evaluation.py` |
| Member 5 | `notebooks/demo.ipynb` (Phase 1 only) | **Placeholder:** `src/cost_analysis.py`; **planned:** `notebooks/05_cost_analysis.ipynb`, `tests/test_cost_analysis.py` |

Member 1 coordinates shared project configuration (`pyproject.toml`, `uv.lock`,
`requirements.txt`), methodology (`PROJECT_RESEARCH_PLAN.md`), integration QA and
shared documentation. The Logistic Regression factory's primary development owner
is Member 2; Member 1 retains responsibility for initial-baseline integration and
must coordinate factory maintenance with Member 2. Likewise, Member 4 owns evaluation
changes and must preserve/coordinate the interface consumed by Member 1's training
workflow. These are explicit collaboration boundaries, not duplicate module owners.

Changes to shared data utilities, preprocessing, interfaces or configuration require
Member 1 coordination/review. Members must not independently modify another owner's
module without agreement. Preserve valid existing implementations and tests.
Training CV schemes and tuning budgets for Members 2 and 3, and the hypothetical
cost units/ratio orientation for Member 5, require team agreement before experiments;
this assignment document does not prescribe those experimental choices.

## Shared boundaries and handoff order

1. Member 1 completes the official data integration, EDA, reviewed shared splits,
   train-only preprocessing and initial baselines, including persistence checks.
   Record real-data limitations honestly until authentication and execution succeed.
2. The validated foundation is available on `dev`; Member 1 reviews integration
   and coordinates further shared-code changes.
3. Members 2 and 3 develop models **in parallel**, using the same manifest,
   feature order and interfaces. They hand pipelines, configurations, seeds,
   dependency versions and validation predictions to Member 4.
4. Member 4 develops independent evaluation/calibration/explainability utilities
   and unit tests immediately, then integrates actual model outputs when available.
5. Member 5 develops independent cost-analysis utilities and unit tests immediately,
   then integrates actual predictions, Member 4's outputs and final results into
   the demonstration. Member 1 coordinates one final held-out test evaluation only
   after model and threshold decisions are locked.

These are artifact dependencies, not a requirement that all five members work
sequentially. Independent utilities may use clearly labeled synthetic unit fixtures;
reported research metrics and findings must use the official dataset.

## Development and Pull Request workflow

1. Fetch the remote branches and switch to your assigned existing branch.
2. Run `uv sync --locked` and implement your assigned functionality.
3. Add/update relevant tests and commit completed changes on that feature branch.
4. Fetch and rebase your feature onto the latest `origin/dev`.
5. Resolve conflicts deliberately, run `uv run pytest`, and push your feature.
6. Open a Pull Request with **base `dev`** for Member 1's integration review.

Example after your assigned branch is already checked out:

```powershell
uv sync --locked
# Implement your task and tests; inspect and stage only the intended files.
git status
git add <reviewed-files>
git commit -m "Describe the completed change"
git fetch origin
git rebase origin/dev
uv run pytest
git push
```

Keep the working tree clean before rebasing. For conflicts, inspect `git status`,
resolve markers with the module owner, stage resolved files and run
`git rebase --continue`. Use `git rebase --abort` if unsure. Never use
`git reset --hard` to discard uncommitted work.

If rebasing rewrites already-pushed commits, coordinate with anyone sharing the
branch and use `git push --force-with-lease` **only on your own feature branch**.
Never force-push or rebase shared `main` or `dev` history. Never overwrite another
member's branch or create a new feature from `main`/another feature.

Each PR must identify behavior changes, affected shared interfaces/configurations,
exact test commands/results, experiment configuration/seed and any real-data
limitations. Include reviewed summaries or artifact references, not credentials,
raw data or bulky fitted models. Merge only after validation; Squash and Merge is
recommended for cohesive feature PRs. Member 1 coordinates reviewed `dev` milestones
into stable `main`. Do not change the GitHub default branch without team agreement.
