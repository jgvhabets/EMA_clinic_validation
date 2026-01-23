

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