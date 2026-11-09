import pytest
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
