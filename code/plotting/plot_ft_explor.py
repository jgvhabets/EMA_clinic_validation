# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt


# current repo imports
from utils.load_utils import get_onedrive_path
from utils.pred_utils import check_skewness

def plot_ft_distribution(
    ft_df, ft_name, EMA_ref,
    sub_id, FT_TYPE, FT_PARAMS_VERSION,
    SES=None, save_plot=False,
):
    
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0] = plot_ft_hist(
        df=ft_df,
        feat_name=ft_name,
        ax=axes[0],
        show=False,
    )

    axes[1] = scatter_feat_vs_EMA(
        df=ft_df,
        feat_name=ft_name,
        EMA_col_name=EMA_ref,
        ax=axes[1],
        show=False,
    )

    axes[2] = boxplot_feat_vs_EMA(
        df=ft_df,
        feat_name=ft_name,
        EMA_col_name=EMA_ref,
        ax=axes[2],
        show=False,
    )

    plt.tight_layout()

    if save_plot:
        lid_fts_figpath = os.path.join(
            get_onedrive_path('emaval_fig'),
            'acc_lid_ft_explore', 
            f'{sub_id}_{SES}_{FT_TYPE}_ft{FT_PARAMS_VERSION}'
        )
        if not os.path.exists(lid_fts_figpath):
            os.makedirs(lid_fts_figpath)
        plt.savefig(os.path.join(lid_fts_figpath,
                                 f'{ft_name}_{FT_PARAMS_VERSION}_explore.png'),
                    facecolor='w', dpi=300,)
        plt.close()

    else:
        plt.show()






def plot_ft_hist(
    df, feat_name,
    show=True, save=False, FIGNAME=None,
    FS=14, ax=None,
    SKEW_THRESHOLD=1.0,
):
    """Plot histogram of a feature.

    If |skewness| > SKEW_THRESHOLD, overlays the log-transformed
    distribution in a contrasting colour on the same axes.
    The log transform shifts the data to be strictly positive before
    applying log1p, so it handles zeros and small negatives.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        return_ax = False
    else:
        return_ax = True

    vals = df[feat_name].dropna().values.astype(float)
    is_skewed, skewness = check_skewness(vals, SKEW_THRESHOLD)

    ax.hist(vals, bins=20, color='olive', alpha=0.7,
            label=f'raw  (skew={skewness:.2f})')

    if is_skewed:
        # shift so all values are > 0, then apply log1p
        shift = max(0.0, -np.nanmin(vals)) + 1e-6
        vals_log = np.log10(vals + shift)
        # scale log values to the same x-range as raw for visual comparison
        ax2 = ax.twiny()
        ax2.hist(vals_log, bins=20, color='orange', alpha=0.4,
                 label='log-transformed')
        ax2.set_xlabel(f'log({feat_name})', fontsize=FS - 2, color='orange')
        ax2.tick_params(axis='x', labelcolor='orange')
        # combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2,
                  fontsize=FS - 3, loc='upper right')

    ax.set_xlabel(feat_name, fontsize=FS)
    ax.set_ylabel('Frequency', fontsize=FS)

    if return_ax:
        return ax

    if save and FIGNAME:
        plt.savefig(FIGNAME)

    if show:
        plt.show()


def scatter_feat_vs_EMA(
    df, feat_name, EMA_col_name,
    show=True, save=False, FIGNAME=None,
    JIT_SIZE=.3, FS=14, ax=None,

):
    x_jitter = np.random.uniform(low=-JIT_SIZE, high=JIT_SIZE, size=len(df))
    y_jitter = np.random.uniform(low=-JIT_SIZE, high=JIT_SIZE, size=len(df))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        return_ax = False
    else:
        return_ax = True

    ax.scatter(
        df[EMA_col_name] + x_jitter,
        df[feat_name] + y_jitter,
        color='olive', alpha=.7,
        
    )

    ax.set_xlabel('EMA score', fontsize=FS,)
    ax.set_ylabel(feat_name, fontsize=FS,)

    if return_ax:
        return ax

    if save and FIGNAME:
        plt.savefig(FIGNAME)

    if show:
        plt.show()


def boxplot_feat_vs_EMA(
    df, feat_name, EMA_col_name,
    show=True, save=False, FIGNAME=None,
    FS=14, ax=None,
):
    ft_values = df[feat_name]
    ema_values = df[EMA_col_name]
    drop_nans = ~np.isnan(ft_values) & ~np.isnan(ema_values)
    ft_values = ft_values[drop_nans]
    ema_values = ema_values[drop_nans]
    ema_categs = sorted(ema_values.unique())
    groups = [ft_values[ema_values == s].values for s in ema_categs]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        return_ax = False
    else:
        return_ax = True

    ax.boxplot(groups, tick_labels=[str(int(s)) for s in ema_categs],
               patch_artist=True,
               boxprops=dict(facecolor='olive', alpha=0.7),
               medianprops=dict(color='black', linewidth=2))

    ax.set_xlabel('EMA score', fontsize=FS)
    ax.set_ylabel(feat_name, fontsize=FS)

    vals = ft_values.astype(float)
    is_skewed = check_skewness(vals)

    if is_skewed:
        # shift so all values are > 0, then apply log1p
        shift = max(0.0, -np.nanmin(vals)) + 1e-6
        vals_log = np.log10(vals + shift)
        log_groups = [
            vals_log[ema_values == s] for s in ema_categs
        ]
        # scale log values to the same x-range as raw for visual comparison
        ax2 = ax.twinx()
        ax2.boxplot(log_groups, patch_artist=True, label='log-transformed',
                    tick_labels=[str(int(s)) for s in ema_categs],
                    boxprops=dict(facecolor='orange', alpha=0.5),
                    medianprops=dict(color='black', linewidth=2))

        ax2.set_ylabel(f'log({feat_name})', fontsize=FS - 2, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        # # combined legend
        # handles1, labels1 = ax.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(handles1 + handles2, labels1 + labels2,
        #           fontsize=FS - 3, loc='upper right')

    if return_ax:
        return ax

    if save and FIGNAME:
        plt.savefig(FIGNAME)

    if show:
        plt.show()