# Team contributor workflow

Use an existing assigned feature branch. Members do not create their own branches.
Member 1 must first confirm all five branches have been published at the same dev
commit. If your remote branch is missing, ask Member 1 to finish setup; do not
create it from main or another feature branch.

## Install, clone and select your branch

Install uv (Windows PowerShell; reopen the terminal if PATH has not refreshed):

```powershell
winget install --id=astral-sh.uv -e
uv --version
git clone https://github.com/kimhoangthien/CO3135-261-ProgrammingAI-DS.git
cd CO3135-261-ProgrammingAI-DS
git fetch origin
git branch -r
```

Choose exactly one first-checkout command:

| Member | First checkout |
|---|---|
| Member 1 | git switch --track origin/feature/data-pipeline |
| Member 2 | git switch --track origin/feature/baseline-models |
| Member 3 | git switch --track origin/feature/xgboost-imbalance |
| Member 4 | git switch --track origin/feature/evaluation-explainability |
| Member 5 | git switch --track origin/feature/cost-analysis-demo |

If your branch already exists locally, use git switch feature/<assigned-name>
instead. Inspect git status before switching; commit completed work or deliberately
stash it and restore it on the right branch. Never discard it with reset --hard.

```powershell
uv sync --locked
uv run pytest
uv run python -m ipykernel install --sys-prefix --name credit-risk --display-name "Credit Risk (uv)"
uv run jupyter lab
```

Python is pinned to 3.12.7. uv creates .venv without requiring manual activation.
Select Credit Risk (uv) in Jupyter, or .venv/Scripts/python.exe in VS Code. Check
sys.executable points to the repository's .venv. pyproject.toml and uv.lock are
authoritative. requirements.txt is an optional generated compatibility export;
never edit it independently. Keep dependency changes separate and coordinated.

## Official data and shared workflow

Create Kaggle credentials at https://www.kaggle.com/settings/api and store them
outside the repository, for example %USERPROFILE%/.kaggle/access_token. Sign into
the same account and join/accept the GiveMeSomeCredit competition rules. See
README's Official data section for legacy credentials and environment alternatives.

From the repository root:

```powershell
uv run python scripts/download_dataset.py
# Running again validates the existing CSV and skips downloading.
uv run python scripts/download_dataset.py
uv run python -m nbconvert --to notebook --execute notebooks/01_eda.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output 01_eda.executed.ipynb --output-dir artifacts
uv run python -m src.split
uv run python -m src.train
uv run pytest
```

EDA and baselines require the real data/raw/cs-training.csv. Authentication errors
are blockers, not evidence about dataset characteristics. Do not publish fixture
metrics as research results. No raw datasets, credentials or model artifacts go
into Git. Notebook output is saved to ignored artifacts; do not commit sensitive
or bulky executed notebooks. EDA charts/tables go to results/figures and tables.

Member 1 creates and reviews data/processed/splits.json. Download the identical
ordered official CSV; reuse the reviewed manifest (transfer it through the team's
approved private artifact channel or create it once with the exact agreed
data/seed). Check its fingerprint with src.split.load_splits; mismatches must stop
experiments. Do not edit/redefine shared splits. `python -m src.split` validates
and reuses an existing manifest, and creates one only if absent.
`python -m src.train` requires an existing valid manifest. Replacing an existing manifest
requires explicit team approval and `python -m src.split --force`; do not
overwrite it with new ratio/seed settings. Keep seed 42 and stratified ratios
70/15/15.

Use src.data.FEATURES in order. Fit src.preprocessing.build_model_pipeline only
on training rows; transform validation without refitting. Never impute, scale or
resample the full dataset. Tree models can use scale=False; Logistic Regression
uses scale=True. Optional class weights apply during fitting. Future SMOTE belongs
inside training folds, never on holdouts. No automatic removal of extreme values.

## Model handoff

Follow model_contract.md: fit(X_train,y_train), predict_proba(X)[:,1], classes [0,1].
Class 1 is serious delinquency within two years, not legal default/financial loss.
Persist the entire fitted preprocessing plus estimator pipeline and verify loaded
probabilities match originals. Never load untrusted pickle/joblib files.

Initial baseline outputs:

- artifacts/baselines/: dummy_prior.joblib, logistic_regression.joblib,
  configuration.yaml, splits.json and dependencies.txt.
- results/tables/baselines/: metrics.json and per-model validation_predictions.csv.

Prediction columns: sample_id,y_true,y_pred_proba,model_name,split. sample_id is
the stable zero-based row position in the cleaned ordered raw dataframe, excluded
from features. Keep configuration, seed, feature order, dataset fingerprint,
Git revision, dependency versions and the split manifest with every fitted model.

Baseline threshold is fixed at 0.5, with no tuning. Metrics distinguish probability
scores (ROC-AUC, PR-AUC as average precision, Brier) from threshold precision/recall/
F1/confusion matrix (rows=true, columns=predicted, order [0,1]). Balanced Logistic
Regression probabilities are uncalibrated; compare Brier/calibration carefully.
The held-out test is reserved for final evaluation. No test-set selection, threshold
search, calibration fitting or resampling during development.

## Rebase, conflicts and pull requests

Commit completed work on your assigned feature branch. Keep the tree clean, then:

```powershell
git fetch origin
git rebase origin/dev
uv run pytest
git push
```

If conflicts occur, inspect git status, edit conflict markers deliberately, stage
the resolved files with git add <files>, and run git rebase --continue. Repeat as
needed and rerun tests after the rebase. Use git rebase --abort to return safely
if unsure; coordinate shared-code conflicts with the owner/Member 1. Do not use
reset --hard or discard collaborators' work.

If rebasing rewrote commits already pushed, use git push --force-with-lease only
on your own feature branch, after coordinating anyone sharing it. Never force-push
main or dev, and never rebase those shared branches.

Open each feature PR with **base=dev**, never main. Include a concrete behavior
summary, exact validation commands/results, config/seed changes, schema impacts,
and honest data availability/experiment limitations. Attach small reviewed tables
or private artifact references rather than raw data or model binaries. Confirm
no credentials or generated data are staged. Member 1 reviews integration; merge
only after validation. Squash and Merge is recommended for cohesive feature PRs.
Promote reviewed dev milestones to main separately; do not change GitHub's default
branch or repository protections without team approval.
