"""Member 5: hypothetical validation costs, no model fitting or test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import yaml
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config, resolve
from src.cost_analysis import threshold_cost_curve, validate_prediction_inputs
from src.data import load_data
from src.train import shared_splits

BASE_MODELS = ("logistic_tuned", "random_forest_tuned", "xgboost_standard_tuned")
RATIOS = (1, 3, 5, 10)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_analysis():
    cfg = load_config()
    frame = load_data()
    parts, manifest = shared_splits(frame, cfg)
    upstream_meta = yaml.safe_load(resolve("artifacts/04_evaluation/configuration.yaml").read_text(encoding="utf-8"))
    fingerprint = json.loads(manifest.read_text())["fingerprint"]
    if (upstream_meta["dataset_fingerprint"] != fingerprint
            or upstream_meta["test_evaluated"] is not False
            or any(upstream_meta["selected_calibration_by_model"].get(name) != "isotonic" for name in BASE_MODELS)):
        raise ValueError("Member 4 provenance or selected calibration differs from audited inputs")
    source = resolve(cfg["outputs"]["tables"]) / "04_evaluation"
    names = [name + suffix for name in BASE_MODELS for suffix in ("", "__isotonic")]
    paths = {name: source / f"{name}_validation_predictions.csv" for name in names}
    predictions = validate_prediction_inputs(
        {name: pd.read_csv(path) for name, path in paths.items()},
        parts["validation"], frame.iloc[parts["validation"]][cfg["target"]].to_numpy(),
    )
    tables = resolve(cfg["outputs"]["tables"]) / "05_cost_analysis"
    figures = resolve(cfg["outputs"]["figures"]) / "05_cost_analysis"
    if tables.exists() or figures.exists():
        raise FileExistsError("05_cost_analysis outputs already exist; move them aside before generating new results")
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    rows, sampled = [], []
    for name, data in predictions.items():
        base = name.removesuffix("__isotonic")
        calibration = "isotonic" if name.endswith("__isotonic") else "uncalibrated"
        for ratio in RATIOS:
            curve = threshold_cost_curve(data.y_true, data.y_pred_proba, ratio, 1)
            best = curve.loc[curve.total_cost.idxmin()]
            common = dict(model_name=name, base_model=base, calibration=calibration,
                          c_fn=ratio, c_fp=1, cost_ratio=f"{ratio}:1", split="validation")
            for decision, point in [("validation_optimum", best),
                                    ("all_negative", curve.iloc[0]),
                                    ("all_positive", curve.iloc[-1])]:
                values = point.to_dict()
                for key in ["tn", "fp", "fn", "tp"]: values[key] = int(values[key])
                rows.append({**common, "decision": decision, **values})
            # Fixed 0.5 comparison uses the same >= convention.
            pred = data.y_pred_proba.to_numpy() >= .5
            y = data.y_true.to_numpy()
            tn, fp = int(((y==0)&~pred).sum()), int(((y==0)&pred).sum())
            fn, tp = int(((y==1)&~pred).sum()), int(((y==1)&pred).sum())
            cost = ratio*fn+fp
            rows.append({**common, "decision":"fixed_0.5", "threshold":.5,
                         "tn":tn,"fp":fp,"fn":fn,"tp":tp,
                         "total_cost":cost,"cost_per_observation":cost/len(y)})
            indices = np.unique(np.r_[np.linspace(0,len(curve)-1,min(500,len(curve))).astype(int),best.name])
            sample = curve.iloc[indices].copy()
            for key,value in common.items(): sample[key] = value
            sampled.append(sample)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "cost_comparison.csv",index=False)
    sensitivity = pd.concat(sampled,ignore_index=True)
    sensitivity.to_csv(tables / "threshold_sensitivity_sampled.csv",index=False)
    selected = comparison[comparison.decision.eq("validation_optimum")]
    selected.to_csv(tables / "selected_thresholds.csv",index=False)
    paired = selected.pivot(index=["base_model","c_fn","c_fp"],columns="calibration",values="total_cost")
    paired["isotonic_minus_uncalibrated_cost"] = paired.isotonic-paired.uncalibrated
    paired.to_csv(tables / "calibration_cost_comparison.csv")
    fig,axes = plt.subplots(2,2,figsize=(13,9))
    for ax,ratio in zip(axes.flat,RATIOS):
        for name in names:
            subset = sensitivity[sensitivity.model_name.eq(name)&sensitivity.c_fn.eq(ratio)].sort_values("threshold")
            ax.plot(subset.threshold,subset.cost_per_observation,label=name,
                    linestyle="-" if name.endswith("__isotonic") else "--")
        ax.set(title=f"Hypothetical C_FN:C_FP = {ratio}:1",xlabel="Threshold (positive if p >= threshold)",ylabel="Validation cost / observation")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Development results: validation-selected costs; sampled curve display")
    fig.tight_layout(); fig.savefig(figures / "threshold_cost_sensitivity.png",dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,6))
    for name in names:
        subset=selected[selected.model_name.eq(name)].sort_values("c_fn")
        ax.plot(subset.c_fn,subset.cost_per_observation,marker="o",label=name)
    ax.set(xlabel="Hypothetical C_FN (C_FP = 1)",ylabel="Minimum validation cost / observation",title="Validation-selected development costs")
    ax.set_xticks(RATIOS); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "cost_ratio_sensitivity.png",dpi=160); plt.close(fig)
    def git(*args):
        result=subprocess.run(["git",*args],cwd=ROOT,capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    provenance_paths = {**paths,"split_manifest":manifest,
                        "member4_configuration":resolve("artifacts/04_evaluation/configuration.yaml"),
                        "cost_source":ROOT/"src/cost_analysis.py", "runner_source":Path(__file__)}
    metadata=dict(analysis_name="05_cost_analysis",created_utc=datetime.now(timezone.utc).isoformat(),
        hypothetical_costs=True,cost_ratio_orientation="C_FN:C_FP",ratios=list(RATIOS),
        selection="minimum empirical validation cost; exact ties choose highest threshold",
        prediction_rule="probability >= threshold",all_negative_threshold=float(np.nextafter(1.,np.inf)),
        validation_rows=len(parts["validation"]),test_evaluated=False,models=names,
        interpretation="Validation-selected development results, not independent generalization or actual monetary losses",
        shap_model="uncalibrated xgboost_standard_tuned; does not explain calibrated ensemble",
        split_fingerprint=json.loads(manifest.read_text())["fingerprint"],
        git_revision=git("rev-parse","HEAD"),git_status=git("status","--porcelain"),
        python=platform.python_version(),dependencies={p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","matplotlib"]},
        inputs={key:dict(path=str(path.resolve()),sha256=sha256(path)) for key,path in provenance_paths.items()},
        figures=str(figures),tables=str(tables),sensitivity_export="At most 500 score-boundary points plus optimum per curve; optimization uses every boundary")
    (tables/"provenance.json").write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    print(selected[["model_name","cost_ratio","threshold","total_cost","cost_per_observation"]].to_string(index=False))
    print(f"Outputs: {tables} and {figures}")
    return tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_analysis()


if __name__ == "__main__": main()
