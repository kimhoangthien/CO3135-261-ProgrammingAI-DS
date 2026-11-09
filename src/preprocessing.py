"""Factories only: callers fit on shared training rows, never on holdouts."""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessor(scale=False, add_indicator=True):
    steps = [('imputer', SimpleImputer(strategy='median', add_indicator=add_indicator,
                                     keep_empty_features=True))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)

def build_model_pipeline(estimator, scale=False, add_indicator=True, balanced=False):
    """Optional class_weight avoids resampling; reject unsupported estimators.

    Set estimator random_state explicitly before passing it here.
    """
    if balanced:
        if 'class_weight' not in estimator.get_params():
            raise ValueError('Estimator does not support class_weight; configure training weights separately')
        estimator.set_params(class_weight='balanced')
    return Pipeline([('preprocessing', build_preprocessor(scale, add_indicator)),
                     ('model', estimator)])
