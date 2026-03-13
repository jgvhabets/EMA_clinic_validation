
from scipy.stats import skew
import numpy as np

from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """Drop features whose Pearson |r| with an earlier feature exceeds threshold.

    Fit computes the correlation matrix on training X and records which
    columns to keep. Transform applies the same column mask, so train/test
    are always treated identically — no data leakage.

    Parameters
    ----------
    threshold : float, default 0.9
        Features with |r| >= threshold to any earlier kept feature are dropped.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X.T))  # shape (n_feats, n_feats)
        keep = []
        for j in range(corr.shape[1]):
            # drop column j if it correlates too strongly with any kept column
            if not any(corr[k, j] >= self.threshold for k in keep):
                keep.append(j)
            # else:
            #     print(f"Dropping feature {j} due to correlation with kept features.")
        self.keep_cols_ = keep
        return self

    def transform(self, X, y=None):
        return X[:, self.keep_cols_]



def prepare_Xy_for_regression(
    X, y=None, remove_nan_rows=True, return_nanrow_bool=False,
    transform_skewed_feats=True, skew_threshold=1.5,
    categorize_y=None,
    verbose=False,
):
    """
    categorize_y requires dict with keys for new scores, and 
    values with list of old scores to be categorized together, e.g.:
    categorize_y = {1: [1, 2, 3], 2: [4, 5], 3: [6, 7, 8, 9]}
    """
    if verbose: print('pre prep X:', X.shape)
    if verbose and y is not None: print('pre prep y:', y.shape)

    ### only nan-removal here; scaling happens within prediction pipe
    if remove_nan_rows:
        nan_rows = np.any(np.isnan(X), axis=1)
        X = X[~nan_rows]
        if y is not None:
            y = y[~nan_rows]

    ### correct skweness with log-transform if needed
    if transform_skewed_feats:
        for i_col in range(X.shape[1]):
            col_values = X[:, i_col]
            X[:, i_col] = logtransform_skewed_feats(col_values, skew_threshold=skew_threshold)


    if verbose: print('post prep X', X.shape)
    if verbose and y is not None: print('post prep y', y.shape)
    # check for nans
    if verbose: print('any nans in X?', np.any(np.isnan(X)))
    if verbose and y is not None: print('any nans in y?', np.any(np.isnan(y)))

    ### transform y
    if categorize_y is not None and y is not None:
        for new_cat, old_cats in categorize_y.items():
            y[np.isin(y, old_cats)] = new_cat
    
    if y is not None:
        output = [X, y]
        if return_nanrow_bool: output.append(nan_rows)
    else:
        output = [X]
        if return_nanrow_bool: output.append(nan_rows)

    return output


def fit_cv_regr(X_train, y_train):
    
    # Pipeline: scaling + ElasticNet
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10000, random_state=42))
    ])

    # Hyperparameter grid
    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    grid_rg = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_rg.fit(X_train, y_train)

    best_model = grid_rg.best_estimator_

    return best_model


def logtransform_skewed_feats(vals, skew_threshold=1.5):
    """
    log-transform features with skewness above threshold.
    """
    X_transformed = vals.copy()    

    if len(vals.shape) == 2:  # if 2D, apply to each column
        for i in range(vals.shape[1]):
            skew_bool, _ = check_skewness(vals[:, i], threshold=skew_threshold)
            if skew_bool:
                X_transformed[:, i] = log_transf(vals[:, i])  # log(1 + x) to handle
    else:  # if 1D, apply directly
        skew_bool, _ = check_skewness(vals, threshold=skew_threshold)
        if skew_bool:
            X_transformed = log_transf(vals)

    return X_transformed


def log_transf(vals):
    shift = max(0.0, -np.nanmin(vals)) + 1e-6
    vals = np.log10(vals + shift)
    return vals


def check_skewness(vals, threshold=1.0):
    """Check if the skewness of the values exceeds a threshold."""
    skewness = skew(vals)
    skewed_bool = abs(skewness) > threshold

    return skewed_bool, skewness


def get_keys_incl(
    avail_df_keys, excl_hr=True,
    keys_excl = [
        'timestamp',
        'tremor',
        'LID',
        'overall_move',
        'move_hands',
        'med_state',
        'wellbeing',
    ],
):
    if excl_hr:
        keys_incl = [
            k for k in avail_df_keys
            if (k not in keys_excl) and ('hr' not in k)
        ]
    else:
        keys_incl = [
            k for k in avail_df_keys if k not in keys_excl
        ]
    return keys_incl