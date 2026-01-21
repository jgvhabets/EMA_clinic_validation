
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr

from utils.load_utils import get_onedrive_path

def scatter_preds(
    y, y_pred_total,
    ZSCORE_Y=False,
    show=True, save=False, FIGNAME=None,
    JIT_SIZE=.3,
    FS=14,

):
    x_jitter = np.random.uniform(low=-JIT_SIZE, high=JIT_SIZE, size=len(y))
    y_jitter = np.random.uniform(low=-JIT_SIZE, high=JIT_SIZE, size=len(y))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(
        y.ravel() + y_jitter,
        y_pred_total.ravel() + x_jitter,
        s=40, alpha=.3,
    )

    if ZSCORE_Y: TICKS, MIDTICK = [-.5, 0, .5, 1.], 0
    else: TICKS, MIDTICK = [1, 3, 5, 7, 9], 5

    ax.set_xlabel('Reported Dyskinesia\n(z-scored EMA)', size=FS,)
    ax.set_ylabel('Predicted Dyskinesia\n(z-scored EMA)', size=FS,)
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICKS)
    ax.set_yticks(TICKS)
    ax.set_yticklabels(TICKS)
    ax.tick_params(size=FS, labelsize=FS, axis='both',)
    ax.axhline(MIDTICK, color='gray', lw=1, alpha=.75,)
    ax.axvline(MIDTICK, color='gray', lw=1, alpha=.75,)
    for h in TICKS:
        ax.axhline(h, color='gray', lw=1, alpha=.3,)
        ax.axvline(h, color='gray', lw=1, alpha=.3,)

    ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['left', 'bottom']].set_visible(False)

    f_regr_skl = f_regression(y_pred_total.reshape(-1, 1), y.reshape(-1, 1))
    pearson_r = pearsonr(y_pred_total.reshape(-1, 1), y.reshape(-1, 1))

    ax.set_title(
        f'Overall performance:\npearson-R: {round(pearson_r[0][0], 1)}, '
        f'F-stat: {round(f_regr_skl[0][0], 1)}, p = {round(pearson_r[1][0], 5)}',
        size=FS,
    )

    plt.tight_layout()

    if save:
        plt.savefig(
            os.path.join(get_onedrive_path('figures'), f'proof_kin_pred', FIGNAME),
                dpi=300, facecolor='w',
            )

    if show: plt.show()
    else: plt.close()

    