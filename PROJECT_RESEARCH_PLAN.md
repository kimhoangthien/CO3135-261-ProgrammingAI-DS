# Explainable and Cost-Sensitive Credit Risk Scoring Using Machine Learning

Phase 1 establishes data validation, EDA, shared 70/15/15 stratified splits,
and leakage-safe preprocessing. No final models or experiments are trained.
The endpoint is SeriousDlqin2yrs, not legal default or monetary loss.

Proposed parallel ownership for five members: data/EDA and shared splits;
Logistic Regression; Random Forest; XGBoost; calibration/evaluation and later
explanation/cost analysis. The team should approve ownership and WP numbering.
WP2: implement comparable model pipelines and record reproducible configurations.
WP3: compare probability calibration, SHAP explanations and threshold-based costs
with explicitly justified cost assumptions. All models use the same manifest.
See docs/model_contract.md. Lock test evaluation until model selection is complete.
