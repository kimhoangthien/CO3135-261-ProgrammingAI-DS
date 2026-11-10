"""Real-data checks skip explicitly when the official training CSV is unavailable."""
import numpy as np
import pytest
from src.config import load_config, resolve
from src.data import FEATURES, load_data
from src.preprocessing import build_preprocessor
from src.split import make_splits, save_splits, load_splits


@pytest.mark.skipif(not resolve(load_config()['dataset_path']).is_file(), reason='Official Kaggle training CSV unavailable')
def test_official_shared_data_contract(tmp_path):
    cfg = load_config()
    frame = load_data()
    parts = make_splits(frame, cfg['target'], cfg['split_ratios'], cfg['random_seed'])
    assert parts == make_splits(frame, cfg['target'], cfg['split_ratios'], cfg['random_seed'])
    flat = sum(parts.values(), [])
    assert sorted(flat) == list(range(len(frame)))
    path = tmp_path / 'splits.json'
    save_splits(frame, parts, path)
    assert load_splits(frame, path) == parts
    overall = frame[cfg['target']].mean()
    for ids in parts.values():
        assert abs(frame.iloc[ids][cfg['target']].mean()-overall) < .01
    # Test partition is checked for coverage/class proportions only; no test features are transformed.
    train = frame.iloc[parts['train']][FEATURES]
    validation = frame.iloc[parts['validation']][FEATURES]
    prep = build_preprocessor(scale=True)
    prep.fit(train)
    np.testing.assert_allclose(prep.named_steps['imputer'].statistics_, train.median().fillna(0))
    result = prep.transform(validation)
    assert result.shape[1] == len(prep.get_feature_names_out())
    assert np.isfinite(result).all()
