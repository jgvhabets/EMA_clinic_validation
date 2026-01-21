

import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr



def classif_home_ema(
    X, y,
    PERMS=False, N_PERMS=0,
    CLS_METHOD = 'lasso',
    n_fold_cv = 4,
    ZSCORE_Y=False,
    verbose=False,
):
    
    
    CLS_LIB = {
        'linreg': LinearRegression(),
        'lda': LDA(),  # only applicable on categorical or binary values
        'lasso': Lasso(alpha=0.3)
    }
    # StratifiedKFold not possible due to continuous values after zscoring
    skf = KFold(n_splits=n_fold_cv, random_state=27, shuffle=True,)
    skf.get_n_splits()


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

        for i_fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

            clf = CLS_LIB[CLS_METHOD]

            X_train, y_train = X[train_idx], y[train_idx]
            # leave first iteration of permutation unshuffled
            if PERMS and i_iter > 0: np.random.shuffle(y_train)
            
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X[test_idx])
            y_true = y[test_idx]
            
            if CLS_METHOD == 'linreg': y_pred = y_pred.ravel()
            y_pred_total[test_idx] = y_pred

            if not PERMS:
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
            print(f'\tPearson-R (skl): {round(pearson_r[0][0], 1)}, '
                    f'p = {round(pearson_r[1][0], 5)}\n\n')


    if not PERMS: return y_pred_total
    else: return perm_true_results, perm_results