
"""Random Forest baseline and hyperparameter tuning."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ..preprocessing import build_model_pipeline


def build_random_forest_baseline(seed=42, **parameters):
    """Build Random Forest with the shared preprocessing pipeline."""
    estimator = RandomForestClassifier(
        random_state=seed,
        **parameters,
    )

    return build_model_pipeline(
        estimator,
        scale=False,
        add_indicator=True,
    )


def tune_random_forest(X_train, y_train, seed=42, cv=3):
    """Tune Random Forest using training data only."""
    pipeline = build_random_forest_baseline(
        seed=seed,
        n_jobs=1,
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
        "model__min_samples_leaf": [2, 5],
        "model__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    return search
