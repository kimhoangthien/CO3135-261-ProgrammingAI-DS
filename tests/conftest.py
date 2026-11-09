import numpy as np
import pandas as pd
import pytest
from src.data import FEATURES

@pytest.fixture
def frame():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    df['SeriousDlqin2yrs'] = np.tile([0]*9+[1], 20)
    return df
