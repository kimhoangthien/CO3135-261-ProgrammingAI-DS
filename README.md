
# Explainable and Cost-Sensitive Credit Risk Scoring

Phase 1 provides data validation, exploratory analysis, reproducible shared splits,
and preprocessing for future Logistic Regression, Random Forest and XGBoost comparisons.
The official training data has been analyzed, and initial Dummy and Logistic
Regression validation baselines are available locally. SeriousDlqin2yrs measures
serious delinquency, not legal default, bankruptcy or realized loss.

## Setup with uv (PowerShell, repository root)

Python is pinned to **3.12.7**. The existing contourpy 1.4.0 requires Python >=3.12;
using 3.11 would require a downgrade, so this migration preserves the working Python.
`pyproject.toml` is the dependency source of truth and `uv.lock` locks resolution.
Existing package versions are preserved using migration constraints. The default
`dev`, `notebook`, and `download` groups include pytest, JupyterLab/ipykernel, and
Kaggle. XGBoost, SHAP and imbalanced-learn are installed for future work; this does
not implement or train models.

If you already have a pip `.venv`, follow the isolation section below before the
first `uv sync`; it must not overwrite an unvalidated environment.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (validated with 0.13.7):

```powershell
winget install --id=astral-sh.uv -e
# Open a new terminal after installation if uv is not on PATH.
uv --version
uv sync --locked
uv run python --version
uv run python -m ipykernel install --sys-prefix --name credit-risk --display-name "Credit Risk (uv)"
uv run jupyter lab
```

uv uses the project's `.venv` by default; activation is unnecessary. In JupyterLab,
select **Credit Risk (uv)**; in VS Code select `.venv/Scripts/python.exe` or the
same kernel. Verify `sys.executable` points into this repository's `.venv`.
The kernel is registered inside the environment, not in global user configuration.
Re-register it after recreating or relocating an environment.

For an existing pip environment, first validate in isolation (do not run a default
sync over the old environment before validating):

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-uv-validation"
uv sync --locked
uv run pytest -q
# Remove this override to return to the normal project environment:
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

The migration retains the original pip environment as `.venv-pip-backup` after
staging passes. To roll back, close Jupyter and Python processes, rename the uv
`.venv` to an unused backup name, and rename `.venv-pip-backup` back to `.venv`.
Use `.venv/Scripts/python.exe` directly for the old environment; `uv run` would
sync it again. Original versions are recorded in `docs/environment-before-uv.txt`.
The old environment's scripts are usable after restoring its original path.
Do not delete the backup until the team accepts the migration.

uv is the only project environment and dependency manager. `requirements.txt` is
the single generated compatibility export, including ML, development, notebook
and Kaggle download dependencies. It supports Colab/course submission and external
tools requiring this format; it is not an independent dependency source or used
by project setup. Normal setup and recovery use `uv sync --locked`.

Regenerate it after an intentional dependency change:

```powershell
uv export --locked --all-groups --format requirements-txt --output-file requirements.txt
```

The export includes environment markers and hashes. Colab or other hosted
runtimes must meet the Python 3.12 project constraint and use the exported versions;
a different Python or preinstalled package stack is not validated by this migration.

## Official data

The official Kaggle CLI is in the default `download` group:

```powershell
uv sync --locked
```

Configure authentication following the [official Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication):
open [API settings](https://www.kaggle.com/settings/api), generate a token and save
it to `%USERPROFILE%\.kaggle\access_token`. Alternatively create a Legacy API Key
and save the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`.
Environment variables `KAGGLE_API_TOKEN` or the pair `KAGGLE_USERNAME`/`KAGGLE_KEY`
are also supported by the CLI. Keep credentials outside the repository; never
paste them into source code or commit them.

Sign in with the same account, join [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/rules)
and accept its competition rules. Then run from the repository root:

```powershell
uv run python scripts/download_dataset.py
# Explicitly replace a valid local download:
uv run python scripts/download_dataset.py --force
```

The script invokes `kaggle competitions download -c GiveMeSomeCredit`, checks the
ZIP and extracts the official files into `data/raw/`. It validates
`cs-training.csv` with the shared loader before publishing files. Existing valid
training data skips the download unless `--force` is provided; invalid local data
triggers a fresh download. Download, extraction or schema failures preserve the
existing training CSV and remove temporary files. Authentication, competition
access, network and corrupt-archive failures return exit code 1 with setup or retry
instructions. Raw CLI diagnostics are suppressed to avoid printing credentials.
The archive is not retained after extraction.

Manual fallback: download and extract `cs-training.csv` from the competition Data
tab to `data/raw/`. No substitute dataset is provided. The loader removes unnamed
index columns and reports duplicates and missingness without deleting observations.
After a successful download, execute shared splits and the EDA commands below;
reload the manifest with `src.split.load_splits` to validate coverage and fingerprint.
No final models are trained by these steps.

## Execute from repository root

```powershell
uv run python -m src.split
uv run python -m nbconvert --to notebook --execute notebooks/01_eda.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output 01_eda.executed.ipynb --output-dir artifacts
uv run python -m nbconvert --to notebook --execute notebooks/demo.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output demo.executed.ipynb --output-dir artifacts
uv run pytest -q
```

Use the `credit-risk` kernel when opening either notebook in Jupyter or VS Code.
Run notebook cells with the repository as cwd; hosted environments must meet the setup constraints above.
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
- `scripts/download_dataset.py`: official Kaggle download, safe extraction and validation.
- `pyproject.toml`, `uv.lock`, `.python-version`: dependency groups, lock and interpreter pin.
- `requirements.txt`: single generated compatibility export, including Kaggle.
- `src/train.py`, `src/evaluate.py`, `src/models/logistic.py`: minimal validation-only baselines.
- Remaining experiment modules: documented future scope for Members 2-5.
- `notebooks/01_eda.ipynb`, `demo.ipynb`: runnable real-data analysis and demonstration.
- `docs/`: data dictionary, honest EDA status, shared model contract.
- `tests/`: synthetic unit fixtures only; no claimed dataset findings.
- `data/`, `artifacts/`, `results/`: local inputs and ignored generated outputs.

## Git and parallel work

`main` holds the validated foundation; `dev` integrates reviewed features.
All five feature branches must be created independently from the same dev commit,
after dev is pushed and synchronized. Feature PRs target dev. Never force-push
main/dev or create a feature from main/another feature. Existing branches must
be inspected and preserved. Remote setup is complete only after push and
remote-tip verification succeed.

See [team assignments](docs/TEAM_TASKS.md) and the [contributor workflow](docs/CONTRIBUTING.md)
for each member's exact existing-branch checkout, uv setup, rebase/conflict policy,
file ownership and handoff order. Member 1 leads integration review.

## Status and next work

**Phase 2A status (2026-11-13): real-data EDA and initial validation baselines
are present locally.** The official training CSV contains 150,000 rows and the
local EDA tables, 70/15/15 seed-42 split manifest, baseline metrics, validation
predictions and fitted pipelines are available as ignored local outputs. See
`docs/eda_findings.md` for measured results and caveats. Earlier
Kaggle-authentication blockers in `docs/validation.md` are historical, not the
current workspace state. Test-set evaluation remains reserved for the final
locked workflow.
WP2: implement and compare model pipelines on shared splits after review.
WP3: calibration, SHAP, and cost-sensitive thresholds after models and assumptions
are approved. WP numbering and proposed five-member ownership need team agreement;
no experimental results or financial cost assumptions are invented here.


## Minimal baseline workflow

After configuring Kaggle authentication and accepting the rules:

```powershell
uv sync --locked
uv run python scripts/download_dataset.py
uv run python scripts/download_dataset.py
uv run python -m ipykernel install --sys-prefix --name credit-risk --display-name "Credit Risk (uv)"
uv run python -m nbconvert --to notebook --execute notebooks/01_eda.ipynb --ExecutePreprocessor.kernel_name=credit-risk --ExecutePreprocessor.timeout=600 --output 01_eda.executed.ipynb --output-dir artifacts
uv run python -m src.split
uv run python -m src.train
uv run pytest
```

`configs/baselines.yaml` defines a prior-based DummyClassifier, balanced Logistic
Regression (C=1, lbfgs, max_iter=2000), and a fixed 0.5 decision threshold. No tuning
is performed. Training reuses an existing fingerprint-validated manifest, fits
preprocessing on training only, evaluates validation only, and verifies model
save/reload consistency. Outputs and their schema are documented in CONTRIBUTING.
No test feature transformation or model evaluation occurs. PR-AUC is reported as
average precision; probability and threshold metrics are grouped separately.
Class weighting changes probability interpretation; these are uncalibrated baselines.
No Random Forest/XGBoost, calibration, SHAP or cost experiment is implemented.

These commands require the official CSV. `src.split` validates and reuses an
existing manifest; it creates one only when missing. Use `--force` to replace an
existing manifest only after explicit team approval. `src.train` refuses to run
without a previously validated manifest and never evaluates test rows. The
optional real-data integration test skips clearly when the CSV is absent. Shared
model and split contracts are unchanged.
