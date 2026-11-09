
# Explainable and Cost-Sensitive Credit Risk Scoring

Phase 1 provides data validation, exploratory analysis, reproducible shared splits,
and preprocessing for future Logistic Regression, Random Forest and XGBoost comparisons.
No final models are trained. SeriousDlqin2yrs measures serious delinquency, not legal
default, bankruptcy or realized loss.

## Setup (PowerShell, from repository root)

Python 3.11 or 3.12 is recommended. Exact direct dependency versions are pinned; `requirements-lock.txt` records the complete
validated Python 3.12 Windows environment. Use that lock file for exact reproduction.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip freeze > artifacts/dependency-versions.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name credit-risk --display-name "Credit Risk"
```

## Official data

Sign into [Kaggle Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data),
accept the competition rules, download and extract `cs-training.csv` to
`data/raw/cs-training.csv`. No substitute dataset is provided. Keep raw data local.
The loader removes unnamed index columns, checks numeric features and binary labels,
and reports duplicates and missingness without deleting observations.

## Execute from repository root

```powershell
.\.venv\Scripts\python.exe -m src.split
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute notebooks/01_eda.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output 01_eda.executed.ipynb --output-dir artifacts
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute notebooks/demo.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output demo.executed.ipynb --output-dir artifacts
.\.venv\Scripts\python.exe -m pytest -q
```

Use the `credit-risk` kernel when opening either notebook in Jupyter or VS Code.
In Colab, install requirements and run notebook cells with the repository as cwd.
`configs/default.yaml` centralizes paths, seed 42, 70/15/15 split ratios, planned
model names and metrics. Split CLI accepts `--config path/to/config.yaml`.
EDA writes CSV tables and PNG figures under configured results directories.
Saved split indices refer to the ordered cleaned raw dataframe; fingerprints
reject a changed dataset. All future experiments must reload the same manifest.
Preprocessing factories are unfitted; fit on training only. Class weighting is
supported for compatible estimators. No resampling is performed.

## Layout

- `src/data.py`, `split.py`, `preprocessing.py`: working shared infrastructure.
- `src/config.py`: repository-relative configuration and path resolution.
- `src/models/` and remaining experiment modules: documented future scope.
- `notebooks/01_eda.ipynb`, `demo.ipynb`: runnable real-data analysis and demonstration.
- `docs/`: data dictionary, honest EDA status, shared model contract.
- `tests/`: synthetic unit fixtures only; no claimed dataset findings.
- `data/`, `artifacts/`, `results/`: local inputs and ignored generated outputs.

## Git and parallel work

`main` is stable; `dev` integrates reviewed work. Create feature branches from dev,
e.g. `feature/data-eda`, `feature/logistic-regression`, `feature/random-forest`,
`feature/xgboost`, and `feature/evaluation`. Merge reviewed features into dev and
promote reviewed milestones to main. This repository started with an unborn main;
dev cannot be created until an authorized initial commit exists. After that commit:

```powershell
git switch -c dev
git switch -c feature/your-task
```

No commits or pushes are performed by setup. Generated data, models, virtual
environments and caches are ignored. See `docs/model_contract.md` for fit/probability
interfaces, prediction schema, complete pipeline persistence and experiment metadata.

## Status and next work

Foundation implemented. Real-data execution and findings require the Kaggle CSV.
See `docs/validation.md` for commands actually executed during setup.
WP2: implement and compare model pipelines on shared splits after review.
WP3: calibration, SHAP, and cost-sensitive thresholds after models and assumptions
are approved. WP numbering and proposed five-member ownership need team agreement;
no experimental results or financial cost assumptions are invented here.
