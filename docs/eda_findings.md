# EDA findings

## Data and reproducibility

The local official `cs-training.csv` was loaded after removal of its unnamed
index column. It contains 150,000 rows, ten numeric features and the binary
`SeriousDlqin2yrs` target. Target counts are 139,974 class 0 (93.316%) and
10,026 class 1 (6.684%). The EDA quality report records 609 duplicate
feature/target rows; observations are retained. The data are imbalanced, and
duplicate observations may cross partitions.

The shared manifest uses seed 42 and stratified 70/15/15 proportions. Its
fingerprint is `749ec0258bc5cf33375b61742d626c109dbda73d05428267d7575224a5b21dd7`.
It partitions 105,000 training rows, 22,500 validation rows and 22,500 test rows.
Validation and test each contain 20,996 class-0 and 1,504 class-1 rows. Baseline
development uses training and validation only; test remains reserved.

## Missingness and investigation flags

`MonthlyIncome` has 29,731 missing values (19.821%) and
`NumberOfDependents` has 3,924 (2.616%). The other features and target have no
missing values in the quality tables. The current preprocessing uses training-set
median imputation and missingness indicators; it does not delete observations.

EDA tables flag 269 rows with each of the delinquency features at 96 or 98,
35,137 `DebtRatio` values above one, 3,321 utilization values above one, and one
zero-age value. These are investigation candidates, not proof of invalid records.
The negative-value table reports zero negative values for all ten features.
IQR flags are descriptive and must not be interpreted as errors. No clipping,
sentinel recoding, row removal or feature changes are applied.

The notebook exports descriptive tables and five figures under `results/`.
Generated tables, figures, split manifests and model artifacts are local outputs
and are ignored by Git; the checked-in notebook contains the analysis code. Re-run
the notebook using the repository instructions to regenerate the outputs.

## Initial validation baselines

The saved validation results use threshold 0.5. Dummy prior has ROC-AUC 0.500,
average precision 0.0668 and Brier score 0.0624. Balanced Logistic Regression
has ROC-AUC 0.8021, average precision 0.3136, Brier score 0.1794, precision
0.1824 and recall 0.6682. These are initial validation-only results, not a
test-set evaluation or calibrated probabilities. Class weighting affects
probability interpretation; do not compare the Brier score without this caveat.

Artifacts record the feature order, dataset fingerprint, configuration,
dependency versions, split manifest and a successful pipeline reload probability
check. The full fitted pipelines and exact-schema validation predictions are
available in the ignored local `artifacts/baselines/` and
`results/tables/baselines/` directories.

## Limitations

These are descriptive findings for the local dataset currently available in the
workspace. They do not establish causality, data-entry errors, financial loss,
or model suitability in production. Do not select or tune models, preprocessing,
or thresholds on the test partition. Any changes to the agreed feature set,
duplicate handling, target, or split require evidence and team approval.
