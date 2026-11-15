
"""Tune Logistic Regression using the shared training and validation splits."""

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.config import ROOT, load_config, resolve
from src.data import FEATURES, load_data
from src.evaluate import evaluate_probabilities, validation_predictions
from src.models.logistic import tune_logistic
from src.split import fingerprint
from src.train import shared_splits, verify_preprocessing


def main():
    cfg = load_config()

    # 1. Load official data and validated shared splits.
    frame = load_data(cfg["dataset_path"], cfg["target"])
    splits, manifest_path = shared_splits(frame, cfg)

    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]

    X_train = train[FEATURES]
    y_train = train[cfg["target"]]

    X_val = validation[FEATURES]
    y_val = validation[cfg["target"]]

    # 2. Tune using training data only.
    print("Starting Logistic Regression tuning...")

    search = tune_logistic(
        X_train,
        y_train,
        seed=cfg["random_seed"],
        cv=3,
    )

    model = search.best_estimator_

    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected class ordering [0, 1]")

    preprocessing = verify_preprocessing(model, X_train, X_val)

    # 3. Evaluate on validation only, threshold fixed at 0.5.
    probabilities = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_probabilities(
        y_val,
        probabilities,
        threshold=0.5,
    )

    # 4. Use separate output folders to protect Member 1's artifacts.
    artifacts_dir = resolve(cfg["outputs"]["artifacts"]) / "member2"
    tables_dir = resolve(cfg["outputs"]["tables"]) / "member2"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 5. Persist the entire fitted pipeline.
    model_path = artifacts_dir / "logistic_tuned.joblib"
    joblib.dump(model, model_path)

    # Reload only the artifact created by this trusted local run.
    loaded_model = joblib.load(model_path)

    np.testing.assert_allclose(
        loaded_model.predict_proba(X_val)[:, 1],
        probabilities,
        rtol=1e-12,
        atol=1e-12,
    )

    # 6. Export validation predictions with stable sample IDs.
    predictions = validation_predictions(
        splits["validation"],
        y_val,
        probabilities,
        "logistic_tuned",
    )

    predictions.to_csv(
        tables_dir / "logistic_tuned_validation_predictions.csv",
        index=False,
    )

    # 7. Save metrics and CV results.
    results = {
        "best_params": search.best_params_,
        "cv_average_precision": float(search.best_score_),
        "validation_metrics": metrics,
        "preprocessing": preprocessing,
        "reload_verified": True,
    }

    (tables_dir / "logistic_tuned_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # 8. Save experiment metadata.
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    metadata = {
        "configuration": cfg,
        "model_name": "logistic_tuned",
        "estimator_parameters": model.named_steps["model"].get_params(
            deep=False
        ),
        "search_parameters": {
            "cv": 3,
            "scoring": "average_precision",
            "param_grid": search.param_grid,
            "best_params": search.best_params_,
        },
        "seed": cfg["random_seed"],
        "features": FEATURES,
        "dataset_fingerprint": fingerprint(frame),
        "git_revision": (
            revision.stdout.strip()
            if revision.returncode == 0
            else None
        ),
        "python": platform.python_version(),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }

    (artifacts_dir / "logistic_tuned_configuration.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )

    (artifacts_dir / "splits.json").write_bytes(
        manifest_path.read_bytes()
    )

    versions = sorted(
        f'{d.metadata["Name"]}=={d.version}'
        for d in importlib.metadata.distributions()
    )

    (artifacts_dir / "dependencies.txt").write_text(
        "\n".join(versions) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
