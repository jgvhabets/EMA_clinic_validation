"""
scripts for statistical analysis
"""

import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM


def calc_expl_variances(fitted_model):

    # Fixed-effects variance
    yhat_fixed = fitted_model.model.exog @ fitted_model.fe_params.values
    sigma_f2 = np.var(yhat_fixed, ddof=1)

    # Random-effects variance (τ00 for intercept)
    tau00 = np.asarray(fitted_model.cov_re)[0, 0]          # variance of random intercepts
    sigma_r2 = tau00

    # Residual variance
    sigma_e2 = fitted_model.scale

    R2_m = sigma_f2 / (sigma_f2 + sigma_r2 + sigma_e2)
    R2_c = (sigma_f2 + sigma_r2) / (sigma_f2 + sigma_r2 + sigma_e2)

    return R2_m, R2_c


def run_mixEff_wGroups(dep_var, indep_var,
                       groups, TO_ZSCORE=False,
                       ALPHA=.01,
                       RETURN_CI=False,
                       RETURN_GRADIENT=False,
                       allow_lm_error: bool = True,
                       PRINT_RESULTS=False,
                       ):
    """
    # tests sign effect of LID on ephys
    # Model: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html
    # Results: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html

    Returns:
        - output_list: contains fixed-effect Coeff, pvalue,
            if defined also Conf-Interv, coef-gradient
    """

    # z-score ephys values on group level for scaling
    if TO_ZSCORE:
        dep_var = (dep_var - np.std(dep_var)) / np.mean(dep_var)

    # define model
    lm_model = MixedLM(
        endog=dep_var,  # dependent variable (ephys score)
        exog=indep_var,  # independent variable (i.e., LID presence, movement)
        groups=groups,  # subjects
        exog_re=None,  # (None)  defaults to a random intercept for each group
    )
    # run and fit model
    try:
        lm_results = lm_model.fit()
    except:
        if allow_lm_error:
            return False
        else:
            print(dep_var.shape, indep_var.shape, groups.shape)
            lm_results = lm_model.fit()

    if PRINT_RESULTS:
        print(lm_results.summary())


    # extract results
    fixeff_cf = lm_results._results.fe_params[0]
    pval = lm_results._results.pvalues[0]

    output_list = [fixeff_cf, pval]  # to keep output number dynamic

    if RETURN_CI:
        conf_int = lm_results.conf_int(alpha=ALPHA)[0]
        output_list.append(conf_int)
    
    if RETURN_GRADIENT:
        grad = lm_results._results.params_object()
        output_list.append(grad)

    # print(f'fixed effect coeff: {fixeff_cf}')  # fixed-effect coeffs
    # print(f'Confidence Interval (alpha: {ALPHA}): {conf_int} (p = {pval.round(5)})')
    # print(lm_results.summary())

    # return two, three, or four values
    return output_list
    