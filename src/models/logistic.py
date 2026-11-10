"""Member 1's minimal baseline; Member 2 can reuse this factory for tuning."""
from sklearn.linear_model import LogisticRegression
from ..preprocessing import build_model_pipeline


def build_logistic_baseline(seed=42, **parameters):
    estimator = LogisticRegression(random_state=seed, **parameters)
    return build_model_pipeline(estimator, scale=True, add_indicator=True)
