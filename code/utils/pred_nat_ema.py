

import numpy as np

from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr



def classif_home_ema(
    X, y, day_codes=None,
    PERMS=False, N_PERMS=0,
    CLS_METHOD = 'lasso',
    n_fold_cv = 4,
    leave_day_out_cv=False,
    ZSCORE_Y=False,
    verbose=False,
):
    
    
    CLS_LIB = {
        'linreg': LinearRegression(),
        'lda': LDA(),  # only applicable on categorical or binary values
        'lasso': Lasso(alpha=0.3)
    }
    # StratifiedKFold not possible due to continuous values after zscoring
    if not leave_day_out_cv:
        cvMethod = KFold(n_splits=n_fold_cv, random_state=27, shuffle=True,)
        cvMethod.get_n_splits()
        fold_params = {'X': X, 'y': y}

    else:
        cvMethod = LeaveOneGroupOut()
        cvMethod.get_n_splits(groups=day_codes)
        fold_params = {'X': X, 'y': y, 'groups': day_codes}

    y_pred_total = np.zeros_like(y).ravel()

    if PERMS:
        np.random.seed(27)
        n_iters = N_PERMS + 1
        perm_results = {'F': [], 'R': [], 'p': []}
    else:
        n_iters = 1

    if X.shape[0] < X.shape[1]: X = X.T


    ##### prediction
    for i_iter in np.arange(n_iters):
        if verbose: print(f'iteration {i_iter}')

        for i_fold, (train_idx, test_idx) in enumerate(cvMethod.split(**fold_params)):
            # print(f'fold # {i_fold}, test size: {len(test_idx)}')

            clf = CLS_LIB[CLS_METHOD]

            X_train, y_train = X[train_idx], y[train_idx]
            # leave first iteration of permutation unshuffled
            if PERMS and i_iter > 0: np.random.shuffle(y_train)
            
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X[test_idx])
            y_true = y[test_idx]
            
            if CLS_METHOD == 'linreg': y_pred = y_pred.ravel()
            y_pred_total[test_idx] = y_pred

            if not PERMS and not leave_day_out_cv:
                f_regr_skl = f_regression(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
                pearson_r = pearsonr(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
                
                if verbose:
                    print(f'Fold # {i_fold}:')
                    print(f'\tF-stat (skl): {round(f_regr_skl[0][0], 1)}, '
                            f'p = {round(f_regr_skl[1][0], 5)}')
                    print(f'\tPearson-R (skl): {round(pearson_r[0][0], 1)}, '
                            f'p = {round(pearson_r[1][0], 5)}\n\n')


        ### MANUAL OUTLIER CORRECTION
        if CLS_METHOD == 'linreg':
                if ZSCORE_Y: y_pred_total[y_pred_total > 3] = 2
                else: y_pred_total[y_pred_total > 12] = 10


        ### CALCULATE TOTAL CROSSVAL PERFORMANCE
        f_regr_skl = f_regression(y_pred_total.reshape(-1, 1), y.reshape(-1, 1))
        pearson_r = pearsonr(y_pred_total.reshape(-1, 1), y.reshape(-1, 1))

        if PERMS:
            if i_iter == 0:
                perm_true_results = {
                        'F': f_regr_skl[0][0],
                        'R': pearson_r[0][0],
                        'p': pearson_r[1][0]
                }
            else:
                perm_results['F'].append(f_regr_skl[0][0])
                perm_results['R'].append(pearson_r[0][0])
                perm_results['p'].append(pearson_r[1][0])
        
        if verbose:
            print(f'TOTALLL CV')
            print(f'\tF-stat (skl): {round(f_regr_skl[0][0], 1)}, '
                    f'p = {round(f_regr_skl[1][0], 5)}')
            print(f'\tPearson-R (skl): {round(pearson_r[0][0], 2)}, '
                    f'p = {round(pearson_r[1][0], 5)}\n\n')
            # print(f'full pearson-R: {pearson_r}')


    if not PERMS: return y_pred_total
    else: return perm_true_results, perm_results



def prep_X_y(
    X, y,
    REMOVE_ZERO_SUBMOVES=False,
    RETURN_EXCLUDED_NAN_BOOL=False,
    RETURN_ZSCORE_PARAMS=False,
    USE_GIVEN_ZSCORE_PARAM=False,
    ZSCORE_PARAMS=None,
    verbose=False,
):
    
    nan_rows = np.any(np.isnan(X), axis=1)

    if REMOVE_ZERO_SUBMOVES:
        zero_rows = np.any(X == 0.0, axis=1)
        nan_rows = np.logical_or(nan_rows, zero_rows)

    if verbose: print(f'removed window-rows bcs of NaNs: n={sum(nan_rows)}')
    X = X[~nan_rows]

    # double check nan removing
    check_nan_rows = np.any(np.isnan(X), axis=1)
    if verbose: print(f'after removing n-nanrows: n={sum(check_nan_rows)}')

    if RETURN_ZSCORE_PARAMS: z_param_list = []
    
    # z-score and log
    for i_col in np.arange(X.shape[1]):
        x_temp = X[:, i_col]
        if USE_GIVEN_ZSCORE_PARAM:
            x_temp = normalize_values(
                x_temp, ZSCORE=True, LOG=True,
                SET_Z_mean_std=ZSCORE_PARAMS[i_col],
            )
        else:
            x_temp, z_params = normalize_values(
                x_temp, ZSCORE=True, LOG=True,
                RETURN_Z_PARAMS=True
            )
            if RETURN_ZSCORE_PARAMS: z_param_list.append(z_params)
        X[:, i_col] = x_temp


    # get array with length n-windows
    if len(y) > 0: y = y[~nan_rows]
    else: y = None

    if RETURN_EXCLUDED_NAN_BOOL and RETURN_ZSCORE_PARAMS:
        return X, y, nan_rows, z_param_list
    elif RETURN_EXCLUDED_NAN_BOOL:
        return X, y, nan_rows
    elif RETURN_ZSCORE_PARAMS:
        return X, y, z_param_list
    else:
        return X, y
    


def normalize_values(
    values, ZSCORE=True, LOG=True,
    return_kept_idx=False,
    SET_Z_mean_std=None,
    RETURN_Z_PARAMS=False,
):
    """
    order of return output:
        values, kept_idx, (zmean, zstd)

    """

    # log transform    
    if LOG:
        # remove zeros and nans
        if return_kept_idx:
            kept_idx = np.where([~np.isnan(v) and v != 0.0 for v in values])[0]
        values = [v for v in values if ~np.isnan(v) and v != 0.0]
        values = np.log(values)
        
    # zscore
    if ZSCORE:
        if SET_Z_mean_std:
            zmean, zstd = SET_Z_mean_std
        else:
            zmean, zstd = np.nanmean(values), np.nanstd(values)
        values = (values - zmean) / zstd
    
    if return_kept_idx and RETURN_Z_PARAMS:
        return values, kept_idx, (zmean, zstd)
    elif return_kept_idx:
        return values, kept_idx
    elif RETURN_Z_PARAMS:
        return values, (zmean, zstd)
    else:
        return values