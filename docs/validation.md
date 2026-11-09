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
