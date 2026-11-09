"""Load raw observations without statistical transformations or row removal."""
import warnings
import numpy as np
import pandas as pd
from .config import load_config, resolve

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
]

def validate_data(frame, target='SeriousDlqin2yrs'):
    missing = set(FEATURES + [target]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Dataset is empty')
    if frame[target].isna().any() or not frame[target].isin([0, 1]).all():
        raise ValueError('Target must contain only non-missing binary labels 0 and 1')
    if frame[target].nunique() != 2:
        raise ValueError('Dataset must contain both target classes')
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES):
        raise ValueError('All features must be numeric')
    if np.isinf(frame[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Features contain infinite values')

def quality_report(frame):
    return {'duplicate_records': int(frame.duplicated().sum()),
            'missing_counts': frame.isna().sum().to_dict()}

def load_data(path=None, target=None):
    cfg = load_config()
    path = resolve(path or cfg['dataset_path'])
    if not path.is_file():
        raise FileNotFoundError(
            f'Dataset not found: {path}. Download cs-training.csv from '
            'https://www.kaggle.com/competitions/GiveMeSomeCredit/data '
            'after signing in and accepting the competition rules; place it in data/raw/.')
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.str.match(r'^Unnamed:')]
    validate_data(frame, target or cfg['target'])
    report = quality_report(frame)
    if report['duplicate_records'] or frame.isna().any().any():
        warnings.warn(f'Data quality report (rows retained): {report}', UserWarning)
    return frame
