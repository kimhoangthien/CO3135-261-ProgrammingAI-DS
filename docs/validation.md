> Current setup: use the uv commands in README.md. Earlier sections below are
> historical execution records, not current installation instructions. References
> to requirements-lock.txt describe the former export, removed during finalization.

# Phase 1 validation ? 2026-11-09

Environment: Windows, Python 3.12. Pinned direct requirements installed successfully
using `.venv/Scripts/python.exe -m pip install -r requirements.txt`.
Full resolved versions are recorded in `requirements-lock.txt`.

Executed checks:

- `.venv/Scripts/python.exe -m pytest -q`: **17 passed in 4.10s** outside the sandbox.
- First full sandboxed run: 12 passed, 5 setup errors, 3 cache warnings. Errors were
  Windows access-denied failures in pytest's temporary directory; rerun passed all.
- An earlier test invocation before installation finished could not import pytest.
- `.venv/Scripts/python.exe -m pip check`: no broken requirements found.
- Both notebook files validated with `nbformat.validate`.
- Source/test files and notebook code cells parsed successfully with Python AST.
- `git check-ignore` confirmed raw CSV, model artifact, virtual environment and
  generated figure paths are ignored.
- `git diff --check` passed (new project files were untracked).
- `.venv/Scripts/python.exe -m src.split`: exited 1 with the expected explicit
  FileNotFoundError and Kaggle instructions because the official CSV is absent.

Real-data EDA execution, generated shared splits, and empirical findings have not
been verified. No replacement dataset was generated. Synthetic data exists only
inside unit-test fixtures. No final Logistic Regression, Random Forest or XGBoost
model has been trained. The synthetic serialization test uses DummyClassifier;
class-weight compatibility is tested using a small synthetic decision tree.

Initial venv bootstrap failed under sandbox permissions; it succeeded on the
approved elevated retry. No commits or pushes were made. Main remains unborn;
dev requires an authorized initial commit first.


## Automated downloader validation ? 2026-11-09

Added scripts/download_dataset.py using official Kaggle CLI 2.2.5, pinned separately
in requirements-download.txt. No Kaggle credential contents were read or logged.
Credential presence checks found no standard kaggle.json/access_token file,
KAGGLE_API_TOKEN, or complete KAGGLE_USERNAME/KAGGLE_KEY environment pair.

- Full final test run: `.venv/Scripts/python.exe -m pytest -q` outside the sandbox:
  **34 passed in 3.08s**. New tests use mocked CLI results and local synthetic ZIPs.
- Initial sandbox run: 12 passed, 21 temporary-directory setup errors, 3 cache
  warnings. Elevated run exposed one error-classification bug (32 passed, 1 failed);
  fixed it and added a CRC-corruption test before the final passing run.
- Live command attempted: `.venv/Scripts/python.exe scripts/download_dataset.py`.
  Initial sandbox attempt hit filesystem permissions. Elevated attempt exited 1
  with the sanitized Kaggle failure message (network/authentication/access checks).
  No archive or training CSV was downloaded. Network availability and competition
  permissions could not be established from that result; credentials are absent
  in the checked standard locations.
- EDA and real-data split workflow were not run because the official CSV is absent.
- Next action: generate a token at https://www.kaggle.com/settings/api and save it
  outside the repository at %USERPROFILE%/.kaggle/access_token (or use legacy
  kaggle.json), accept https://www.kaggle.com/competitions/GiveMeSomeCredit/rules
  with that account, then rerun the downloader and README workflow.

No final models were trained, and no commits or pushes were made by this task.


## uv migration ? 2026-11-10

Result: migration successfully validated in an isolated environment and then at
its final `.venv` path. uv 0.13.7; CPython **3.12.7** (unchanged). Python 3.11 was
not adopted: installed contourpy 1.4.0 declares Requires-Python >=3.12. No downgrade
was forced. Before migration the project used stdlib venv plus pip; the shell had
no active VIRTUAL_ENV and its default python was an unrelated MSYS interpreter.

Inspected README, all three requirements files, `.venv/pyvenv.cfg`, installed
package metadata, Git status, ignore rules and prior validation results before
writing configuration. Pre-existing uncommitted downloader changes were retained.
Original installed versions are recorded in `docs/environment-before-uv.txt`.
Every original installed package remains present at exactly the same version;
there were **no existing-package upgrades, downgrades or removals**.

Direct versions:

| Package | Version | Change |
|---|---|---|
| NumPy | 2.1.3 | Preserved |
| pandas | 2.2.3 | Preserved |
| SciPy | 1.14.1 | Preserved |
| scikit-learn | 1.5.2 | Preserved |
| matplotlib | 3.9.2 | Preserved |
| seaborn | 0.13.2 | Preserved |
| PyYAML | 6.0.2 | Preserved |
| joblib | 1.4.2 | Preserved |
| pytest | 8.3.3 | Preserved; dev group |
| ipykernel | 6.29.5 | Preserved; notebook group |
| nbformat / nbconvert | 5.10.4 / 7.16.4 | Preserved; notebook group |
| jupytext | 1.19.6 | Preserved; notebook group |
| Kaggle | 2.2.5 | Preserved; download group |
| XGBoost | 2.1.4 | Added for required import validation |
| SHAP | 0.46.0 | Added for required import validation |
| imbalanced-learn | 0.12.4 | Added; compatible with NumPy 2 and sklearn 1.5 |
| JupyterLab | 4.3.5 | Added for required notebook workflow |

New transitive dependencies support those four additions and are recorded in
uv.lock. uv installed 126 packages on Windows; the universal lock resolves 131
including platform-specific dependencies. `pyproject.toml` uses package=false,
so the repository structure and root-relative module execution are preserved.
All groups are selected by default. The old requirements files are now generated
exports of uv.lock, including hashes and environment markers, not independent lists.

Executed commands (uv below was invoked as
`.\artifacts\uv-bootstrap\bin\uv.exe`, bootstrapped without altering the old venv):

```powershell
py -3.12 -m pip install --target artifacts/uv-bootstrap uv
.\.venv\Scripts\python.exe -m pytest -q
uv lock
$env:UV_PROJECT_ENVIRONMENT = '.venv-uv-validation'
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python artifacts/validate_uv_environment.py
uv pip check --python .venv-uv-validation/Scripts/python.exe
# After staging passed, original .venv was renamed to .venv-pip-backup.
# A new shell with no UV_PROJECT_ENVIRONMENT override targeted default .venv:
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python artifacts/validate_uv_environment.py
uv pip check --python .venv/Scripts/python.exe
uv export --locked --all-groups --format requirements-txt --output-file requirements.txt --no-emit-project
uv export --locked --all-groups --format requirements-txt --output-file requirements-lock.txt --no-emit-project
uv export --locked --only-group download --format requirements-txt --output-file requirements-download.txt --no-emit-project
```

Results:

- Original pip environment: **34 passed in 11.26s**.
- Isolated uv environment: **34 passed in 2.38s**.
- Final project `.venv`: **34 passed in 9.31s**, including all original 17 tests.
- `uv sync --locked`: succeeded in both clean environments; repeat staging sync
  checked installed packages successfully without changing the lock.
- All requested ML imports passed: pandas, numpy, sklearn, xgboost, shap, imblearn.
- Every baseline installed package/version matched in both uv environments.
- `uv pip check`: all 126 installed packages compatible in both environments.
- Both project notebook schemas validated. A live notebook cell imported all
  requested ML libraries and repository data/split/preprocessing modules from root.
- `credit-risk` kernel was installed using `ipykernel install --sys-prefix`; its
  argv resolves to the correct interpreter in the final `.venv`. Display name:
  **Credit Risk (uv)**. No global user kernelspec was changed.
- JupyterLab started in both environments and its authenticated local HTTP
  `/api/status` endpoint returned **200**. Smoke-test servers were shut down.
- Ignore checks confirmed environments, credentials, datasets and artifacts are
  excluded, while uv.lock is retained. No source/notebook/test/config or model
  contract changes were made by this migration.

Non-final attempts and limitations:

- Sandboxed uv bootstrap/lock/export attempts encountered package lookup or cache
  access restrictions; approved elevated retries succeeded.
- `uv run pytest -q` initially failed to import src. The documented command is
  `uv run python -m pytest -q`, preserving the prior successful invocation; no
  algorithm, test or import-path code was changed.
- The first Jupyter HTTP harness incorrectly compared the Windows launcher PID
  to the child server PID. Logs showed successful server startup; correcting the
  harness and bypassing proxies for localhost yielded the HTTP 200 checks above.
- Windows zmq warned that Proactor uses a selector fallback thread. Kernel code
  executed successfully; no event-loop or package changes were forced.
- Full EDA/demo analysis was not executed: the real dataset is absent. Only
  environment/kernel execution and notebook schemas were validated. No synthetic
  data was presented as real findings; no final models were trained.
- uv is bootstrapped locally under ignored artifacts; install it on PATH using
  the README winget command for the short `uv` commands in normal terminals.

Rollback: the original environment was renamed only after independent validation,
and is retained at `.venv-pip-backup`. Close Python/Jupyter, rename the current
`.venv` to an unused name, then restore `.venv-pip-backup` to `.venv`. Invoke its
Python directly instead of uv run. Keep the backup until team acceptance. The
ignored staging environment also remains available. No commits or pushes made.

Final entry-point checks: uv run --locked python scripts/download_dataset.py --help and uv run --locked python -m src.split --help both exited 0. Migration-only Jupyter runtime endpoints were checked during cleanup; no test server remained accessible.


## uv finalization ? 2026-11-10 (current workflow)

uv is the single primary dependency/environment manager. Inspected all repository
files, requirements exports, README, documentation, scripts and notebooks. No CI,
deployment configuration, setup script or course-submission automation requiring
requirements-lock.txt was found. Confirmed all project and dependency-group pins
are present in uv.lock and match installed versions before cleanup.

Changes:

- Removed requirements-lock.txt after confirming its dependency/hash contents
  exactly duplicated requirements.txt (only generated command headers differed).
- Retained requirements.txt as a generated full compatibility export for the
  originally supported Colab/course interchange. External consumers are not
  assumed absent. Retained requirements-download.txt for existing downloader
  compatibility messages/tests. Neither is an independent dependency source;
  regenerate with the uv export commands in README.
- Updated downloader installation-error text to recommend uv sync --locked;
  transfer, authentication, extraction and schema-validation logic are unchanged.
- Added pytest pythonpath=["."] in pyproject.toml so uv run pytest works from root
  without changing tests or application imports. This supersedes the earlier
  migration observation about the pytest console entry point.
- Updated README to use uv run pytest, removed the duplicate-export command,
  documented retained exports, and marked prior validation entries as historical.
- Retained historical pip command records, original installed-version snapshot,
  rollback environment and working environment. No pip installation was removed.

Executed validation (uv invoked via artifacts/uv-bootstrap/bin/uv.exe):

- uv sync --locked: succeeded, checked 126 packages; uv.lock unchanged.
- uv run pytest -q: **34 passed in 3.59s**, including all original 17 tests.
- uv run python artifacts/validate_uv_environment.py: required numpy, pandas,
  sklearn, xgboost, shap and imblearn imports passed; notebook schemas and live
  repository-module execution passed; credit-risk kernel uses .venv's interpreter.
- JupyterLab started under .venv and authenticated /api/status returned HTTP 200.
  The temporary server was shut down after verification.
- Compared hashes before/after: src/, notebooks/, tests/, configs/, research plan,
  model contract and uv.lock unchanged. All installed package versions unchanged.
- .venv, rollback environment, credentials, datasets and models remain ignored.
  No source/notebook currently contains a pip installation instruction. Only
  historical validation records retain commands actually executed in the past.
- git diff --check passed, with Git's existing LF/CRLF normalization notices.

An initial sandboxed sync failed on uv cache permissions; the approved elevated
rerun passed. Windows zmq still emits its non-blocking selector-fallback warning.
The real dataset remains absent, so full EDA was not run during this cleanup.
uv is still locally bootstrapped; install on PATH using the README instructions.
No dependencies upgraded/downgraded, environments deleted, final models trained,
commits created or changes pushed.

## Post-validation cleanup — 2026-11-10

Removed the temporary .venv-uv-validation environment, pytest/Python caches, Jupyter smoke-test runtime files and logs, temporary JSON reports, and the ignored validation helper. Earlier references to those artifacts are historical execution records. Kept the working .venv, .venv-pip-backup rollback environment, and artifacts/uv-bootstrap (the only locally available uv executable). No application code or tests changed; tests were not rerun for cache cleanup. The last full run passed all 34 tests.

## Single requirements export — 2026-11-10

Consolidated compatibility exports into requirements.txt. Verified every package/version in requirements-download.txt was already present in the full uv-generated export before removing the download-only file. Updated README, downloader recovery-message references, and their corresponding test expectation. Dependency configuration, uv.lock and installed versions are unchanged. Executed uv run --locked pytest -q -p no:cacheprovider tests/test_download_dataset.py: 17 passed in 0.53s. Earlier references to requirements-download.txt are historical.


## Phase 2A prerequisite validation ? 2026-11-11

Outcome: **blocked by Kaggle authentication; leader scope is incomplete**.
Inspected the existing data loader, downloader, split implementation,
preprocessing factory, model contract, notebook analysis, locked dependencies,
and current raw-data directory before attempting real-data work.

Executed with local uv at artifacts/uv-bootstrap/bin/uv.exe:

- uv sync --locked: succeeded; resolved 131 packages, checked 126 installed.
- uv run --locked python scripts/download_dataset.py: exited 1 with
  "Kaggle authentication is missing or invalid." Reused the existing official
  downloader; no substitute dataset was generated. No credential values exposed.
- uv run pytest -q -p no:cacheprovider: **34 passed in 7.12s**.
- Initial sandbox attempts were blocked by uv cache permissions; the approved
  elevated run produced the results above. The existing environment and lock
  were preserved, with no dependency changes.

The CSV is absent. Consequently real-data EDA, reloadable shared split creation,
training-only preprocessing integration checks, Dummy/Logistic Regression fitting,
validation metrics/predictions and pipeline artifact reload checks were not run.
The existing 34 unit tests are not proof of successful real-data integration.
No real-data integration tests or baseline implementations were added in this
blocked attempt. No Random Forest/XGBoost, calibration, SHAP or cost experiments
were performed; no commits or pushes were made.

Required user action: create Kaggle credentials at
https://www.kaggle.com/settings/api and save the token outside the repository to
%USERPROFILE%/.kaggle/access_token (or legacy kaggle.json in the same directory).
Join and accept https://www.kaggle.com/competitions/GiveMeSomeCredit/rules with that
account. Run uv run python scripts/download_dataset.py and rerun it to check the
valid-data skip path. Unit download-idempotency tests currently pass, but the
real-data skip path cannot be confirmed until a successful official download.

Handoff order remains: authenticated download; execute/review EDA; create/reload
and validate the unchanged seed-42 70/15/15 shared split; validate train-only
preprocessing and feature consistency; implement/train Dummy and Logistic
Regression baselines; export validation-only metrics/predictions and verify
pipeline reload; then hand the shared manifest and model contract to the other
four members. Keep test rows out of fitting, model selection and threshold tuning.


## Team foundation finalization (2026-11-11)

Audit: main and the live origin/main both pointed to
744c684c80977240838dc191fa6a6cee638e7788 before this foundation update. The three
previously staged status documents were inspected and preserved. No local dev or
feature branch existed; live git ls-remote --heads origin showed only main after
a successful fetch. Earlier remote attempts failed to connect; the final retry
succeeded. No history reset/rewrite was used.

Implementation: completed minimal Dummy prior and balanced Logistic Regression
factories, validation-only training/evaluation, fixed threshold 0.5, full-pipeline
persistence/reload checks, exact-schema prediction exports and experiment metadata.
The training CLI reuses a saved fingerprint-validated split manifest and rejects
conflicting seed/ratio settings. Median/scaler statistics are checked against
training data. Feature definitions, shared split implementation, model contract,
random seed, uv.lock and dependency versions are unchanged. Added fixture tests,
an optional real-data integration test, member assignments and contributor policy.
No Member 2-5 experiment was implemented.

Executed checks (from root):

- uv sync --locked: succeeded, resolved 131 packages, checked 126 installed.
- uv run pytest -q -p no:cacheprovider: **43 passed, 1 skipped in 3.32s**.
  All original tests passed. The skipped real-data test explicitly requires the
  absent official CSV. Fixture metrics/models exist only in pytest temporary
  directories, not as experimental results or committed artifacts.
- Baseline fixture workflow verified training-only fit rows, validation-only
  predict rows, train-derived imputation statistics, exact prediction schema,
  saved/loaded pipeline probability equality, and unchanged manifest on rerun.
- Logistic Regression, Random Forest and XGBoost interface/preprocessing factories
  were checked; no RF/XGBoost estimator was fitted. Probability metrics were
  checked against known expected values. PR-AUC means average precision.
- uv run python artifacts/foundation_environment_check.py: numpy, pandas, sklearn,
  xgboost, shap, imblearn and JupyterLab imported; both notebook schemas validated;
  the live credit-risk kernel used this repository's .venv interpreter and
  imported shared repository modules successfully. Windows zmq emitted its known
  selector-fallback warning, but notebook execution passed.
- uv run python scripts/download_dataset.py: exit 1, Kaggle authentication
  missing/invalid. No credentials found by presence checks; none were printed.
- uv run python -m src.train: exit 1 with the explicit missing official CSV
  message. No real-data artifacts or predictions were produced.

Real-data EDA, real split/preprocessing integration and research baseline metrics
remain blocked on Kaggle setup. Set a token outside Git at
%USERPROFILE%/.kaggle/access_token or configure legacy Kaggle credentials, accept
the competition rules with that account, and follow CONTRIBUTING.md's workflow.
Do not treat passing fixture tests as completed real-data experiments.

Git publication is a separate step after this validation and staged-file review.
The final handoff report records the resulting commit and verified local/remote
branch tips. Only publish after confirming no existing branch will be replaced;
feature creation must return to synchronized dev before each independent branch.
