import pytest
import yaml
import pandas as pd
from src.split import make_splits, save_splits, load_splits

def test_split_contract(frame, tmp_path):
    parts = make_splits(frame)
    assert parts == make_splits(frame)
    assert [len(parts[k]) for k in ['train','validation','test']] == [140,30,30]
    flat = sum(parts.values(), [])
    assert len(set(flat)) == len(frame)
    assert sorted(flat) == list(range(len(frame)))
    for ids in parts.values():
        assert abs(frame.iloc[ids]['SeriousDlqin2yrs'].mean()-.1) <= 1/len(ids)
    path = tmp_path / 'splits.json'
    save_splits(frame, parts, path)
    assert load_splits(frame, path) == parts
    with pytest.raises(ValueError, match='differs'):
        load_splits(frame.iloc[::-1], path)

@pytest.mark.parametrize('ratios', [(1,0,0), (.5,.2,.2), (-.1,.5,.6)])
def test_invalid_ratios(frame, ratios):
    with pytest.raises(ValueError):
        make_splits(frame, ratios=ratios)


def test_manifest_rejects_overlap(frame, tmp_path):
    from src.split import validate_splits
    parts = make_splits(frame)
    parts['test'][0] = parts['train'][0]
    with pytest.raises(ValueError, match='exactly once'):
        validate_splits(frame, parts)


def test_split_cli_reuses_existing_manifest(frame, tmp_path, monkeypatch):
    from src.split import main

    dataset_path = tmp_path / 'dataset.csv'
    frame.to_csv(dataset_path, index=False)
    persisted_frame = pd.read_csv(dataset_path)
    manifest_path = tmp_path / 'processed/splits.json'
    parts = make_splits(persisted_frame)
    save_splits(persisted_frame, parts, manifest_path)
    original_manifest = manifest_path.read_bytes()
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({
        'dataset_path': str(dataset_path),
        'target': 'SeriousDlqin2yrs',
        'random_seed': 42,
        'split_ratios': [0.7, 0.15, 0.15],
        'outputs': {'processed': str(manifest_path.parent)},
    }), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['src.split', '--config', str(config_path)])

    main()

    assert manifest_path.read_bytes() == original_manifest
