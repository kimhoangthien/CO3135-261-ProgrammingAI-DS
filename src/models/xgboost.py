"""XGBoost pipelines and class-imbalance strategies (Member 3)."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

from ..preprocessing import build_model_pipeline

STRATEGIES = ("standard", "weighted", "smote")

# Espace de recherche SANS préfixe ; le préfixe est ajouté selon la stratégie.
XGB_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
}


def compute_scale_pos_weight(y_train):
    """Ratio négatifs / positifs, calculé sur le TRAIN uniquement."""
    y = np.asarray(y_train)
    n_pos = (y == 1).sum()   # nombre de défauts
    n_neg = (y == 0).sum()   # nombre de non-défauts
    if n_pos==0:
        raise ValueError("No positive value")
    return float(n_neg/n_pos)


def build_xgboost_pipeline(seed=42, strategy="standard", scale_pos_weight=None, **params):
    """Pipeline complet : prétraitement partagé + XGBoost (+ SMOTE si demandé)."""
    # 1: vérifier que strategy est dans STRATEGIES
    if strategy not in STRATEGIES:
        raise ValueError("Strategy not accepted")
    # 2: créer XGBClassifier(objective="binary:logistic", tree_method="hist",
    #         random_state=seed, n_jobs=1, **params)
    xgb = XGBClassifier(
        objective="binary:logistic", 
        tree_method="hist",
        random_state=seed, 
        n_jobs=1,
        **params)
    # 3: si "weighted" -> exiger scale_pos_weight et le passer au classifieur
    if strategy == "weighted":
        if scale_pos_weight is None or scale_pos_weight <= 0:
            raise ValueError("strategy='weighted' requires a positive scale_pos_weight")
        xgb.set_params(scale_pos_weight = scale_pos_weight)
    if strategy == "smote":
        estimator = ImbPipeline([
            ("smote", SMOTE(random_state=seed)),
            ("model", xgb),
        ])
    else:
        estimator = xgb
    return build_model_pipeline(estimator, scale=False, add_indicator=True)

def param_prefix(strategy):
    """'model__' ou 'model__model__' selon l'imbrication."""
    if strategy == 'standard' or strategy == 'weighted':
        return 'model__'
    elif strategy == 'smote':
        return 'model__model__'
    raise ValueError("Invalid model name")


def tune_xgboost(X_train, y_train, seed=42, strategy="standard", cv=3,
                 n_iter=20, search_space=None, scale_pos_weight=None):
    """RandomizedSearchCV sur le TRAIN uniquement, scoring average_precision."""
    # 1: construire le pipeline avec build_xgboost_pipeline
    pipeline = build_xgboost_pipeline(seed=seed, strategy=strategy,
                                      scale_pos_weight=scale_pos_weight)
    # 2: préfixer chaque clé de (search_space or XGB_SEARCH_SPACE)
    space = search_space or XGB_SEARCH_SPACE
    prefix = param_prefix(strategy)
    param_distributions = {prefix + name: values for name, values in space.items()}
    # 3: RandomizedSearchCV(pipeline, param_distributions=...,
    #         n_iter=n_iter, scoring="average_precision",
    #         cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
    #         random_state=seed, n_jobs=-1, refit=True)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    # 4: search.fit(X_train, y_train) puis return search
    search.fit(X_train, y_train)
    return search