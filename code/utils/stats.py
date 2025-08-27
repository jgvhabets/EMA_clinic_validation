"""
scripts for statistical analysis
"""

import numpy as np

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
