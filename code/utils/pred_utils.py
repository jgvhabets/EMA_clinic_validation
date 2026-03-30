
from scipy.stats import skew
import numpy as np

from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


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


def remove_outliers_zscore(
    X, y=None, threshold_n_sd=3.0, return_outl_mask=False, verbose=False
):
    """Flag and remove samples where any feature z-score exceeds threshold.

    Parameters
    ----------
    X         : ndarray, already z-scored (output of prepare_X_for_clustering)
    y         : optional array aligned with X rows
    threshold_n_sd : float, default 3.0, number of standard deviations to use as threshold

    Returns
    -------
    X_clean, y_clean (if y provided), outlier_mask (bool, True = outlier)
    """

    outlier_mask = np.any(np.abs(X) > threshold_n_sd, axis=1)

    if verbose:
        print(f'[outliers] {outlier_mask.sum()} / {len(X)} samples removed '
              f'(|z| > {threshold_n_sd})')

    X_clean = X[~outlier_mask]

    output = {'X': X_clean}

    if y is not None: output['y'] = y[~outlier_mask]
    
    if return_outl_mask: output['outl_mask'] = outlier_mask

    return output



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


from itertools import compress


def full_preproc_X_y_regr(
    df,
    EMA_Y,
    EXCL_HR = True,  # exclude heart rate features (not available in all timepoints)
    LOG_SKEWED = True,
    TRANSFORM_Y = None,  # transform LID into 3 classes: 1, 2 (1-4), 3 (>4)
    Z_STD_OUTLIER_THRESH = 3.0,
    APPLY_PCA = False,
    PCA_n_comps = 6,
    apply_trained_pca=None,
    use_skew_feat_bool=None,
    return_skew_feat_bool=False,
    return_trained_pca=False,
    adjust_session_list=None,
):
    # define features to include
    PRED_FTS = get_keys_incl(df.keys(), excl_hr=EXCL_HR,)

    ### define training X, y
    X_all = df[PRED_FTS].values.copy()
    if EMA_Y is not None:
        y_all = df[EMA_Y].values.copy().astype(float)
    else:
        y_all = None

    ### prepare X-data
    if return_skew_feat_bool:
        skew_corr_bool_list = [check_skewness(
            X_all[:, i][~np.isnan(X_all[:, i])],
            threshold=1.0,
        )[0] for i in range(X_all.shape[1])]
    # chek whether there is a pre-defined list of features to log-transform bcs of skewedness
    if isinstance(use_skew_feat_bool, list):
        LOG_SKEWED = False  # perform outside of function

    prep_output = prepare_Xy_for_regression(
        X_all, y_all,
        remove_nan_rows=True,
        return_nanrow_bool=True,
        transform_skewed_feats=LOG_SKEWED,
        verbose=True,
    )
    if EMA_Y is not None: X_all, y_all, nan_rows = prep_output
    else: X_all, nan_rows = prep_output
        
    
    # perform separate log-transform if bool-list given
    if isinstance(use_skew_feat_bool, list):
        for i in range(X_all.shape[1]):
            if use_skew_feat_bool[i]:  # if this feature was log-transformed in training
                X_all[:, i] = log_transf(X_all[:, i])

    # adjust given session ids to nan-rows removed
    if isinstance(adjust_session_list, list):
        adjust_session_list = list(compress(adjust_session_list, ~nan_rows))

        
    ### REMOVE OUTLIERS BASED ON Z SCORE THRESHOLD
    # get mean and std per column and store for later use
    cv_m, cv_sd = np.mean(X_all, axis=0), np.std(X_all, axis=0)
    # column-wise z-score normalization
    X_all = (X_all - cv_m) / cv_sd
    # get mask for rows that contain outliers in any feature (using z-score threshold)
    outlier_mask = np.any(np.abs(X_all) > Z_STD_OUTLIER_THRESH, axis=1)
    # remove outliers from X and y
    X_all = X_all[~outlier_mask]
    if EMA_Y is not None:
        y_all = y_all[~outlier_mask]

    if isinstance(adjust_session_list, list):
        adjust_session_list = list(compress(adjust_session_list, ~outlier_mask))

        
    print(f'X shape after removing outliers (n={sum(outlier_mask)}): {X_all.shape}')

    ### PCA
    if APPLY_PCA:
        if apply_trained_pca is None:
            # fit PCA
            fitted_pca = PCA(n_components=PCA_n_comps).fit(X_all)
        else:
            fitted_pca = apply_trained_pca
        # transform X into components
        X_all = fitted_pca.transform(X_all)
        print(f'X shape after PCA: {X_all.shape}')


    ### prepare y
    if TRANSFORM_Y is not None and EMA_Y is not None:
        for new_y, old_y_list in TRANSFORM_Y.items():
                y_all[np.isin(y_all, old_y_list)] = new_y
    
    output = {}

    if EMA_Y is None:
        output['X_all'] = X_all
    else:
        output['X_all'] = X_all
        output['y_all'] = y_all

    if return_skew_feat_bool:
        output['skew_list'] = skew_corr_bool_list

    if isinstance(adjust_session_list, list):
        output['session_ids'] = np.array(adjust_session_list)
    
    if return_trained_pca:
        output['pca'] = fitted_pca


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


def logtransform_skewed_feats(vals, skew_threshold=1.5,
                              apply_skewed_ft_bool=None,):
    """
    log-transform features with skewness above threshold.
    """
    X_transformed = vals.copy()    

    if len(vals.shape) == 2:  # if 2D, apply to each column
        for i in range(vals.shape[1]):
            if apply_skewed_ft_bool is None:
                skew_bool, _ = check_skewness(vals[:, i], threshold=skew_threshold)
            else:
                skew_bool = apply_skewed_ft_bool[i]
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