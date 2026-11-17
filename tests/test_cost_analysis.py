"""Synthetic cost tests; no research metrics or upstream retraining."""
import numpy as np
import pandas as pd
import pytest
from src.cost_analysis import (threshold_cost_curve, optimal_threshold,
                               validate_costs, validate_prediction_inputs)
from src.evaluate import validation_predictions


def test_exact_sweep_matches_brute_force_with_ties_and_endpoints():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0, 0, .5, .5, 1, 1])
    for ratio in [1, 3, 5, 10]:
        curve = threshold_cost_curve(y, p, ratio, 1)
        for row in curve.itertuples():
            pred = p >= row.threshold
            fp = np.sum((y == 0) & pred)
            fn = np.sum((y == 1) & ~pred)
            assert row.fp == fp and row.fn == fn
            assert row.tp + row.fn == 3 and row.tn + row.fp == 3
            assert row.total_cost == ratio*fn + fp
            assert row.cost_per_observation == row.total_cost / len(y)
        assert curve.iloc[0].tp == 0 and curve.iloc[0].fp == 0
        assert curve.iloc[-1].fn == 0 and curve.iloc[-1].tn == 0


def test_known_optimum_and_cost_orientation():
    y, p = [0, 1, 0, 1], [.1, .4, .6, .9]
    assert optimal_threshold(y, p, 1, 1)["threshold"] == .9
    assert optimal_threshold(y, p, 3, 1)["threshold"] == .4
    assert optimal_threshold(y, p, 3, 1)["total_cost"] == 1


def test_constant_scores_tie_prefers_all_negative_and_is_order_invariant():
    y, p = [0, 1], [.5, .5]
    assert optimal_threshold(y, p)["threshold"] > 1
    assert optimal_threshold(y[::-1], p)["threshold"] > 1


def test_cost_scale_and_zero_costs():
    y, p = [0, 1, 0, 1], [.1, .4, .6, .9]
    a, b = optimal_threshold(y, p, 3, 1), optimal_threshold(y, p, 30, 10)
    assert a["threshold"] == b["threshold"]
    assert b["total_cost"] == 10*a["total_cost"]
    assert optimal_threshold(y, p, 0, 1)["total_cost"] == 0
    assert optimal_threshold(y, p, 1, 0)["total_cost"] == 0


@pytest.mark.parametrize("costs", [(-1,1),(1,-1),(0,0),(np.nan,1),(1,np.inf)])
def test_invalid_costs(costs):
    with pytest.raises(ValueError): validate_costs(*costs)


@pytest.mark.parametrize("y,p", [([0,1],[.2]),([0,1],[np.nan,.2]),
    ([0,1],[-.1,.2]),([0,1],[.1,1.1]),([0,0],[.1,.2]),([],[])])
def test_invalid_probabilities(y,p):
    with pytest.raises(ValueError): threshold_cost_curve(y,p)


def test_alignment_accepts_reordered_rows():
    frame = validation_predictions([7,2],[0,1],[.1,.8],"m")
    aligned = validate_prediction_inputs({"m":frame},[7,2],[0,1])["m"]
    assert aligned.sample_id.tolist() == [2,7]


@pytest.mark.parametrize("change", ["schema","duplicate","labels","split","name","probability","id"])
def test_bad_prediction_inputs(change):
    f = validation_predictions([7,2],[0,1],[.1,.8],"m")
    if change == "schema": f = f.drop(columns="split")
    elif change == "duplicate": f.loc[1,"sample_id"] = 7
    elif change == "labels": f.loc[0,"y_true"] = 1
    elif change == "split": f.loc[0,"split"] = "test"
    elif change == "name": f.loc[0,"model_name"] = "other"
    elif change == "probability": f.loc[0,"y_pred_proba"] = np.nan
    else: f.loc[0,"sample_id"] = 99
    with pytest.raises(ValueError): validate_prediction_inputs({"m":f},[7,2],[0,1])


def test_runner_synthetic_integration_preserves_inputs(tmp_path, monkeypatch):
    import json
    from scripts import run_cost_analysis as runner
    from src.data import FEATURES
    frame = pd.DataFrame({name: np.arange(8,dtype=float) for name in FEATURES})
    frame["SeriousDlqin2yrs"] = [0,1,0,1,0,1,0,1]
    manifest = tmp_path / "data/processed/splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"fingerprint":"synthetic","splits":{"validation":[4,5,6,7]}}))
    source = tmp_path / "results/tables/04_evaluation"
    source.mkdir(parents=True)
    for name in runner.BASE_MODELS:
        for suffix in ["", "__isotonic"]:
            full = name+suffix
            validation_predictions([4,5,6,7],[0,1,0,1],[0,.4,.4,1],full).to_csv(source/f"{full}_validation_predictions.csv",index=False)
    meta = tmp_path / "artifacts/04_evaluation/configuration.yaml"
    meta.parent.mkdir(parents=True); meta.write_text("dataset_fingerprint: synthetic\ntest_evaluated: false\nselected_calibration_by_model:\n  logistic_tuned: isotonic\n  random_forest_tuned: isotonic\n  xgboost_standard_tuned: isotonic\n")
    originals = {p:p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(runner,"load_data",lambda:frame)
    monkeypatch.setattr(runner,"shared_splits",lambda f,c:({"validation":[4,5,6,7]},manifest))
    monkeypatch.setattr(runner,"resolve",lambda path:tmp_path/path)
    tables = runner.run_analysis()
    result = pd.read_csv(tables/"cost_comparison.csv")
    assert len(result)==96
    assert set(result.decision)=={"validation_optimum","all_negative","all_positive","fixed_0.5"}
    for _,group in result.groupby(["model_name","c_fn"]):
        best=group[group.decision.eq("validation_optimum")].total_cost.iloc[0]
        assert (best <= group.total_cost).all()
    provenance=json.loads((tables/"provenance.json").read_text())
    assert provenance["test_evaluated"] is False
    assert len(provenance["models"])==6
    assert all(p.read_bytes()==content for p,content in originals.items())
    with pytest.raises(FileExistsError): runner.run_analysis()
