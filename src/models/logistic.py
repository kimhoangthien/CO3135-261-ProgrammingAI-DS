
"""Logistic Regression baseline and hyperparameter tuning."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from ..preprocessing import build_model_pipeline


def build_logistic_baseline(seed=42, **parameters):
    estimator = LogisticRegression(
        random_state=seed,
        **parameters
    )
    return build_model_pipeline(
        estimator,
        scale=True,
        add_indicator=True
    )


def tune_logistic(X_train, y_train, seed=42, cv=3):
    """Tune Logistic Regression using training data only."""

    pipeline = build_logistic_baseline(
        seed=seed,
        max_iter=2000
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["lbfgs"]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    return search
