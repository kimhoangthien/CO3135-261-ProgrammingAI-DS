# EDA status

The real dataset was not present during initialization. No empirical findings,
figures, class proportions, missingness rates, or anomaly counts are claimed.
Run notebooks/01_eda.ipynb after downloading the official data and review the
exported tables and figures. Values such as delinquency counts 96/98, debt ratios
above one, utilization above one, and age zero are investigation candidates;
they are not automatically invalid and must not be removed without evidence.


## Phase 2A attempt ? 2026-11-11

Status: **blocked before real-data analysis**. The official Kaggle downloader
returned exit code 1: "Kaggle authentication is missing or invalid."
`data/raw/cs-training.csv` is absent. Presence checks found no standard
kaggle.json/access_token file or configured token/legacy environment pair;
no credential contents were read or printed.

No real-data EDA, split generation, baseline fitting, validation metrics,
prediction exports or fitted model artifacts were produced in this attempt.
Dataset size, imbalance, missingness, duplicates, suspicious-value counts and
feature relationships remain unmeasured. Previously listed anomaly candidates
remain investigation questions, not observed findings.

To unblock, create a token at https://www.kaggle.com/settings/api and store it
locally outside the repository at `%USERPROFILE%/.kaggle/access_token`, or create
a legacy API key and save kaggle.json under `%USERPROFILE%/.kaggle/`.
Join https://www.kaggle.com/competitions/GiveMeSomeCredit/rules using that same
account and accept its rules. Then run the README dataset download command.
The leader scope is not complete until the real-data workflow and baselines
specified in Phase 2A are implemented, executed and validated.


## Foundation finalization (2026-11-11)

The official download was retried and again returned "Kaggle authentication is
missing or invalid." No real CSV, real-data EDA outputs, split manifest, research
baseline metrics, fitted research pipelines or prediction exports exist yet.
Minimal baseline training/evaluation code and fixture-only tests are now provided;
this does not complete real-data execution. The EDA notebook now also exports a
data-quality summary including duplicate count when run on the official data.

Preprocessing remains training-only median imputation, optional indicators and
Logistic Regression scaling, with no clipping/deletion or feature changes.
Real-data evidence is required before proposing anomaly treatment. Duplicates can
cross the required stratified partitions; investigate and disclose this risk before
requesting any group-aware split change. Balanced Logistic Regression outputs are
uncalibrated probabilities; interpret calibration/Brier results accordingly.
