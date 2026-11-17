"""Member 3: XGBoost and class-imbalance experiments (E2, E3) on VALIDATION only."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# Allow `python scripts/run_xgboost_experiments.py` from the repository root
# (same pattern as scripts/download_dataset.py).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ROOT, load_config, resolve  # noqa: E402
from src.data import FEATURES, load_data  # noqa: E402
from src.evaluate import evaluate_probabilities, validation_predictions  # noqa: E402
from src.models.xgboost import (build_xgboost_pipeline, compute_scale_pos_weight,  # noqa: E402
                                param_prefix, tune_xgboost)
from src.split import fingerprint  # noqa: E402
from src.train import shared_splits, verify_preprocessing  # noqa: E402


def xgb_params(pipeline):
    """Paramètres du XGBClassifier lui-même, quelle que soit l'imbrication."""
    # TODO: model = pipeline.named_steps["model"] ; si c'est un ImbPipeline
    #       (hasattr(model, "named_steps")), descendre dans model.named_steps["model"]
    #       return xgb.get_params()
    model = pipeline.named_steps["model"]
    xgb = model.named_steps["model"] if hasattr(model, "named_steps") else model
    return xgb.get_params()


def smote_params(pipeline):
    """SMOTE settings for the metadata, or None when the strategy does not resample."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "named_steps"):
        return None
    return model.named_steps["smote"].get_params()


def git_state():
    def run(*command):
        result = subprocess.run(["git", *command], cwd=ROOT, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_revision": run("rev-parse", "HEAD"),
            "git_uncommitted_changes": bool(status) if status is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=None, help="override for a quick smoke run")
    parser.add_argument("--strategies", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config()
    xcfg = load_config(ROOT / "configs/xgboost.yaml")
    seed, threshold = cfg["random_seed"], xcfg["threshold"]
    n_iter = args.n_iter or xcfg["tuning"]["n_iter"]
    strategies = args.strategies or xcfg["strategies"]
    cv = xcfg["tuning"]["cv"]
    smoke_run = n_iter != xcfg["tuning"]["n_iter"]
    if smoke_run:
        print(f"SMOKE RUN: n_iter={n_iter} (config: {xcfg['tuning']['n_iter']}). "
              "Do not report these numbers.", flush=True)

    # 1. Data + shared splits (train / validation only)
    # TODO: frame, splits, manifest ; X_train, y_train, X_val, y_val
    # TODO: spw = compute_scale_pos_weight(y_train)
    frame = load_data(cfg["dataset_path"], cfg["target"])
    splits, manifest_path = shared_splits(frame, cfg)
    # The test partition is never indexed in this script.
    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]
    X_train, y_train = train[FEATURES], train[cfg["target"]]
    X_val, y_val = validation[FEATURES], validation[cfg["target"]]
    spw = compute_scale_pos_weight(y_train)
    print(f"Train {len(X_train)} rows ({int(y_train.sum())} positives), "
          f"validation {len(X_val)} rows; scale_pos_weight={spw:.4f}", flush=True)

    # 2. Output folders
    # TODO: artifacts/03_xgboost et results/tables/03_xgboost (mkdir parents=True, exist_ok=True)
    artifacts_dir = resolve(cfg["outputs"]["artifacts"]) / "03_xgboost"
    tables_dir = resolve(cfg["outputs"]["tables"]) / "03_xgboost"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build the list of (name, strategy, fitted_pipeline, search_or_None)
    models = []
    fit_seconds = {}
    # TODO E3: xgboost_default = build_xgboost_pipeline(seed, "standard").fit(...)
    start = time.perf_counter()
    default = build_xgboost_pipeline(seed=seed, strategy="standard").fit(X_train, y_train)
    fit_seconds["xgboost_default"] = round(time.perf_counter() - start, 1)
    models.append(("xgboost_default", "standard", default, None))
    print(f"[xgboost_default] fitted in {fit_seconds['xgboost_default']}s", flush=True)

    # TODO E2: pour chaque stratégie -> tune_xgboost(..., search_space=xcfg["search_space"],
    #          cv=xcfg["tuning"]["cv"], n_iter=n_iter, scale_pos_weight=spw si weighted)
    #          -> search.best_estimator_ ; enregistrer pd.DataFrame(search.cv_results_) en CSV
    #          afficher le temps écoulé (time.perf_counter) avec print(..., flush=True)
    for strategy in strategies:
        name = f"xgboost_{strategy}_tuned"
        print(f"[{name}] tuning: {n_iter} candidates x {cv} folds ...", flush=True)
        start = time.perf_counter()
        search = tune_xgboost(
            X_train, y_train, seed=seed, strategy=strategy, cv=cv, n_iter=n_iter,
            search_space=xcfg["search_space"],
            scale_pos_weight=spw if strategy == "weighted" else None,
        )
        fit_seconds[name] = round(time.perf_counter() - start, 1)
        cv_results = pd.DataFrame(search.cv_results_).drop(columns="params")
        cv_results.columns = [c.replace("param_" + param_prefix(strategy), "param_")
                              for c in cv_results.columns]
        cv_results.sort_values("rank_test_score").to_csv(
            tables_dir / f"cv_results_{strategy}.csv", index=False)
        models.append((name, strategy, search.best_estimator_, search))
        print(f"[{name}] done in {fit_seconds[name]}s, "
              f"CV average precision={search.best_score_:.4f}", flush=True)

    # 4. Evaluate, persist, export
    metrics, rows, params = {}, [], {}
    for name, strategy, pipeline, search in models:
        # TODO: classes_, verify_preprocessing, proba, evaluate_probabilities
        if list(pipeline.classes_) != [0, 1]:
            raise ValueError(f"{name}: expected class ordering [0, 1]")
        preprocessing = verify_preprocessing(pipeline, X_train, X_val)
        proba = pipeline.predict_proba(X_val)[:, 1]
        evaluation = evaluate_probabilities(y_val, proba, threshold=threshold)

        # TODO: joblib dump/load + np.testing.assert_allclose
        model_path = artifacts_dir / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        np.testing.assert_allclose(joblib.load(model_path).predict_proba(X_val)[:, 1],
                                   proba, rtol=1e-12, atol=1e-12)

        # TODO: validation_predictions(splits["validation"], y_val, proba, name).to_csv(...)
        validation_predictions(splits["validation"], y_val, proba, name).to_csv(
            tables_dir / f"{name}_validation_predictions.csv", index=False)

        # TODO: metrics[name] = {...} ; rows.append({...}) ; params[name] = xgb_params(pipeline)
        prefix = param_prefix(strategy)
        best_params = ({k.removeprefix(prefix): v for k, v in search.best_params_.items()}
                       if search is not None else None)
        cv_score = float(search.best_score_) if search is not None else None
        metrics[name] = {
            "strategy": strategy,
            "tuned": search is not None,
            "best_params": best_params,
            "cv_average_precision": cv_score,
            "validation_metrics": evaluation,
            "preprocessing": preprocessing,
            "fit_seconds": fit_seconds[name],
            "reload_verified": True,
        }
        probability = evaluation["probability_metrics"]
        decision = evaluation["threshold_metrics"]
        (tn, fp), (fn, tp) = decision["confusion_matrix"]
        rows.append({
            "model_name": name,
            "experiment": "E3" if search is None else ("E2+E3" if strategy == "standard" else "E2"),
            "strategy": strategy,
            "tuned": search is not None,
            "cv_average_precision": cv_score,
            "roc_auc": probability["roc_auc"],
            "average_precision": probability["pr_auc_average_precision"],
            "brier_score": probability["brier_score"],
            "threshold": decision["threshold"],
            "precision": decision["precision"],
            "recall": decision["recall"],
            "f1": decision["f1"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fit_seconds": fit_seconds[name],
        })
        params[name] = {"xgboost": xgb_params(pipeline), "smote": smote_params(pipeline)}
        print(f"[{name}] validation ROC-AUC={probability['roc_auc']:.4f} "
              f"AP={probability['pr_auc_average_precision']:.4f} "
              f"Brier={probability['brier_score']:.4f}", flush=True)

    # 5. Summary + metadata
    # TODO: summary.csv, metrics.json (allow_nan=False), configuration.yaml,
    #       copie de splits.json, dependencies.txt (même code que le Membre 2)
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "summary.csv", index=False)
    (tables_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")

    metadata = {
        "configuration": cfg,
        "xgboost_configuration": xcfg,
        "run": {"n_iter": n_iter, "strategies": list(strategies), "cv": cv,
                "smoke_run": smoke_run},
        "seed": seed,
        "scale_pos_weight": spw,
        "features": FEATURES,
        "dataset_fingerprint": fingerprint(frame),
        **git_state(),
        "python": platform.python_version(),
        "library_versions": {pkg: importlib.metadata.version(pkg)
                             for pkg in ["xgboost", "imbalanced-learn", "scikit-learn"]},
        "positive_class": "SeriousDlqin2yrs=1: serious delinquency within two years",
        "estimator_parameters": params,
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    (artifacts_dir / "configuration.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    (artifacts_dir / "splits.json").write_bytes(manifest_path.read_bytes())
    versions = sorted(f'{d.metadata["Name"]}=={d.version}'
                      for d in importlib.metadata.distributions())
    (artifacts_dir / "dependencies.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")

    print()
    print(summary[["model_name", "cv_average_precision", "roc_auc", "average_precision",
                   "brier_score", "precision", "recall", "f1"]].round(4).to_string(index=False))
    print(f"\nOutputs: {artifacts_dir} and {tables_dir}")


if __name__ == "__main__":
    main()
