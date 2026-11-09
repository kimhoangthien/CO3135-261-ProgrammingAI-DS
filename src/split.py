"""Persist positional row identifiers bound to the full ordered dataset."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import load_config, resolve
from .data import load_data

def fingerprint(frame):
    payload = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(str(list(frame.columns)).encode() + payload).hexdigest()

def make_splits(frame, target='SeriousDlqin2yrs', ratios=(.7, .15, .15), seed=42):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('Provide three positive split ratios summing to one')
    indices = np.arange(len(frame))
    train, rest = train_test_split(indices, test_size=ratios[1]+ratios[2],
                                  stratify=frame[target], random_state=seed)
    validation, test = train_test_split(rest, test_size=ratios[2]/(ratios[1]+ratios[2]),
                                      stratify=frame.iloc[rest][target], random_state=seed)
    return dict(train=train.tolist(), validation=validation.tolist(), test=test.tolist())

def save_splits(frame, splits, path, seed=42, ratios=(.7, .15, .15)):
    validate_splits(frame, splits)
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(fingerprint=fingerprint(frame), seed=seed,
                                   ratios=list(ratios), splits=splits), indent=2), encoding='utf-8')

def validate_splits(frame, splits):
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, test splits')
    flat = [i for part in splits.values() for i in part]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(frame))):
        raise ValueError('Splits must cover all rows exactly once')
    if any(not part for part in splits.values()):
        raise ValueError('Splits must be nonempty')

def load_splits(frame, path):
    payload = json.loads(resolve(path).read_text(encoding='utf-8'))
    if payload['fingerprint'] != fingerprint(frame):
        raise ValueError('Dataset content or row order differs from saved splits')
    validate_splits(frame, payload['splits'])
    return payload['splits']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = load_data(cfg['dataset_path'], cfg['target'])
    splits = make_splits(frame, cfg['target'], cfg['split_ratios'], cfg['random_seed'])
    path = resolve(cfg['outputs']['processed']) / 'splits.json'
    save_splits(frame, splits, path, cfg['random_seed'], cfg['split_ratios'])
    print(f'Saved shared splits to {path}: ' + str({k: len(v) for k,v in splits.items()}))

if __name__ == '__main__':
    main()
