# Data dictionary

Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
Definitions summarize the competition variables; verify against the downloaded
Data Dictionary before making substantive research claims.
`SeriousDlqin2yrs` indicates serious delinquency (90 days past due or worse)
within two years. It is not legal default, bankruptcy, or realized financial loss.

| Column | Meaning | Expected type | Missingness | Quality considerations | Suggested preprocessing |
|---|---|---|---|---|---|
| SeriousDlqin2yrs | Serious delinquency endpoint | Binary integer | Reject missing labels | Require both 0 and 1 | Exclude from features |
| RevolvingUtilizationOfUnsecuredLines | Unsecured revolving balance divided by available credit | Continuous numeric | Report actual missingness; retain rows | Ratios above one may reflect over-limit use; investigate | Training-only median imputation, optional indicator; scale for logistic regression |
| age | Borrower age in years | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect zero and extreme ages | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberOfTime30-59DaysPastDueNotWorse | Count of 30â€“59 day delinquency episodes | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect unusually large counts including 96/98 | Training-only median imputation, optional indicator; scale for logistic regression |
| DebtRatio | Monthly debt payments and expenses divided by monthly gross income | Continuous numeric | Report actual missingness; retain rows | Inspect large ratios; denominator and reporting conventions matter | Training-only median imputation, optional indicator; scale for logistic regression |
| MonthlyIncome | Monthly income | Continuous numeric | Report actual missingness; retain rows | Missingness and extreme values require review | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberOfOpenCreditLinesAndLoans | Open installment and revolving credit lines | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect extreme counts | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberOfTimes90DaysLate | Count of episodes 90 or more days past due | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect unusually large counts including 96/98 | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberRealEstateLoansOrLines | Real-estate loans and lines including home equity | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect extreme counts | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberOfTime60-89DaysPastDueNotWorse | Count of 60â€“89 day delinquency episodes | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Inspect unusually large counts including 96/98 | Training-only median imputation, optional indicator; scale for logistic regression |
| NumberOfDependents | Dependents excluding the borrower | Integer-valued numeric (may load as float with missing values) | Report actual missingness; retain rows | Missingness and extreme counts require review | Training-only median imputation, optional indicator; scale for logistic regression |

An unnamed CSV index is excluded from features. Shared sample IDs are positional,
not this source index. Missingness is measured from the supplied file rather than
assumed. No clipping, deletion, or inferred sentinel conversion is applied.
