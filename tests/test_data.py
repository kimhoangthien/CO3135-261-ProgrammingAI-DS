import pytest
from src.data import load_data, validate_data, quality_report

def test_schema(frame):
    validate_data(frame)
    with pytest.raises(ValueError, match='Missing required'):
        validate_data(frame.drop(columns=['age']))

@pytest.mark.parametrize('label', [2, -1, None, 'yes'])
def test_bad_labels(frame, label):
    frame['SeriousDlqin2yrs'] = frame['SeriousDlqin2yrs'].astype(object)
    frame.loc[0, 'SeriousDlqin2yrs'] = label
    with pytest.raises(ValueError, match='binary'):
        validate_data(frame)

def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match='Kaggle|kaggle'):
        load_data(tmp_path / 'missing.csv')

def test_index_and_quality(frame, tmp_path):
    frame.loc[0, 'MonthlyIncome'] = float('nan')
    frame.loc[1] = frame.loc[0]
    path = tmp_path / 'input.csv'
    frame.to_csv(path)
    with pytest.warns(UserWarning, match='rows retained'):
        loaded = load_data(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)
    assert loaded['MonthlyIncome'].isna().sum() == 2
    assert quality_report(loaded)['duplicate_records'] == 1


def test_empty_and_nonnumeric_and_infinite(frame):
    with pytest.raises(ValueError, match='empty'):
        validate_data(frame.iloc[:0])
    changed = frame.copy()
    changed['age'] = 'unknown'
    with pytest.raises(ValueError, match='numeric'):
        validate_data(changed)
    changed = frame.copy()
    changed.loc[0, 'DebtRatio'] = float('inf')
    with pytest.raises(ValueError, match='infinite'):
        validate_data(changed)

def test_single_class(frame):
    frame['SeriousDlqin2yrs'] = 0
    with pytest.raises(ValueError, match='both target classes'):
        validate_data(frame)
