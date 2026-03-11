
from scipy.stats import skew
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def prepare_Xy_for_regression(
    X, y, remove_nan_rows=True, categorize_y=False,
    transform_skewed_feats=True, skew_threshold=1.5,
    verbose=False,
):
    if verbose: print('pre prep', X.shape, y.shape)

    ### only nan-removal here; scaling happens within prediction pipe
    if remove_nan_rows:
        nan_rows = np.any(np.isnan(X), axis=1)
        X = X[~nan_rows]
        y = y[~nan_rows]

    ### correct skweness with log-transform if needed
    if transform_skewed_feats:
        for i_col in range(X.shape[1]):
            col_values = X[:, i_col]
            X[:, i_col] = logtransform_skewed_feats(col_values, skew_threshold=skew_threshold)


    if verbose: print('post prep', X.shape, y.shape)
    # check for nans
    if verbose: print('any nans in X?', np.any(np.isnan(X)))
    if verbose: print('any nans in y?', np.any(np.isnan(y)))

    ### transform y
    if categorize_y:
        y[y == 1] = 1
        y[(y > 1) & (y <= 4)] = 2
        y[y > 4] = 3
    
    return X, y


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