from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    with open(path or ROOT / 'configs/default.yaml', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
