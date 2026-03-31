
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr, mannwhitneyu

from utils.load_utils import get_onedrive_path
from utils.data_handling_ema_acc import get_ft_daily_mean



SES_COLORS = {'ses01': 'violet', 'ses02': 'orange', 'ses03': 'darkcyan'}
SES_LABELS = {'ses01': 'Pre-DBS',
              'ses02': 'Post-DBS',
              'ses03': 'Post-DBS\n(optimized)'}
FONT_SIZES = {
    'title': 16,
    'axes': 14,
    'ticks': 12,
}
EMA_YTICKS = np.arange(1, 10)
EMA_YTICK_LABELS = ['1', '', '3', '', '5', '', '7', '', '9']


def plot_session_boxes2(
    test_pred_EMA, test_sessions,
    ALPHA = .01 / 3,  # Bonferroni correction for three comparisons
    # SES_COLORS=SES_COLORS, SES_LABELS=SES_LABELS, FS=FS,
):
    # plot predicted EMA scores per session as boxplots
    # with significance stars for differences

    avail_sess = np.unique(test_sessions)


    fig, axes = plt.subplots(figsize=(6, 3))

    ax = axes
    bp = ax.boxplot(
        [test_pred_EMA[test_sessions == ses] for ses in avail_sess],
        labels=[SES_LABELS[ses] for ses in avail_sess],
        patch_artist=True
    )
    # make plots pretty
    for patch, ses in zip(bp['boxes'], avail_sess):
        patch.set_facecolor(SES_COLORS[ses])  # color boxes with session colors
        # make median line black and thicker
        median_line = bp['medians'][avail_sess.tolist().index(ses)]
        median_line.set_color('black')
        median_line.set_linewidth(2)
        # make mean line grey and thinner
        ses_mean = np.mean(test_pred_EMA[test_sessions == ses])
        # plot mean as grey line based on calculated means
        ax.plot([avail_sess.tolist().index(ses) + 1 - 0.15,
                avail_sess.tolist().index(ses) + 1 + 0.15],
                [ses_mean, ses_mean],
                color='k', alpha=.5, linestyle='--', linewidth=1)
        # plot mean as grey dot based on calculated means
        ax.plot(avail_sess.tolist().index(ses) + 1, ses_mean, 'o',
                color='k', markersize=5, alpha=.5,)

    # add sgnificance stars based on Mann-Whitney U test results
    for i, ses1 in enumerate(avail_sess):
        for j, ses2 in enumerate(avail_sess):
            if j <= i:  # only compare each pair once and skip self-comparison
                continue
            preds1 = test_pred_EMA[test_sessions == ses1]
            preds2 = test_pred_EMA[test_sessions == ses2]
            stat, p_value = mannwhitneyu(preds1, preds2, alternative='two-sided')
            # if p_value < ALPHA:
            # add line annotation for significant difference w/ transparency
            x1, x2 = i + 1, j + 1
            y, h, col = max(max(preds1), max(preds2)) + 0.5, (i+j)*.25, 'k'
            if p_value < ALPHA: alph = 1
            else: alph = 0.3
            # lines only horizontal between boxes, not vertical to x-axis, and on different y hieghts
            ax.plot([x1, x2], [y + h, y + h], lw=1.5, c=col, alpha=alph,)
            # ax.text((x1+x2)*.5, y + (i+j)*.15 + 0.05, "*", ha='center', va='bottom', color=col)


    # edit axes and ticks
    # ax.set_title('Predicted EMA Scores per Session', fontsize=FS)
    ax.set_ylim(1, 9.1)
    ax.set_yticks(np.arange(1, 10, 2))
    ax.set_ylabel('Predicted Tremor Severity\n(EMA score)', fontsize=FONT_SIZES['axes'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['ticks'],)
    # make upper and right spines invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.show()



def box_fullsession_preds(
    y_preds, session_code, target_EMA_name, ax=None,
    VIOLIN=False, sign_asterix=False,
):
    # create boxplots splitted on session_code
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        return_ax = False
    else:
        return_ax = True

    avail_ses = np.unique(session_code)
    box_data = [y_preds[session_code == ses_id]
                for ses_id in avail_ses]
    positions = [(i*.25) + 1 for i in range(len(avail_ses))]

    if VIOLIN:
        vp = ax.violinplot(box_data, positions=positions,
                           showmedians=False, showextrema=False, widths=0.75)

        # clip each violin to a half: even index → right half, odd index → left half
        for i, (body, ses_id) in enumerate(zip(vp['bodies'], avail_ses)):
            pos = positions[i]
            for path in body.get_paths():
                verts = path.vertices
                if i % 2 == 0:  # right half only
                    verts[verts[:, 0] > pos, 0] = pos
                else:            # left half only
                    verts[verts[:, 0] < pos, 0] = pos
            body.set_facecolor(SES_COLORS[ses_id])
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)

        # overlay boxplot lines
        ax.boxplot(box_data, positions=positions, widths=0.1,
                   patch_artist=False, manage_ticks=False,
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.2),
                   capprops=dict(color='black', linewidth=1.2),
                   boxprops=dict(color='black', linewidth=1.2),
                   flierprops=dict(marker='o', markersize=3,
                                   markeredgecolor='black', alpha=0.5))
    else:
        # boxplot
        bp = ax.boxplot(box_data, positions=positions, widths=0.75,
                        patch_artist=True, manage_ticks=False,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2),
                        boxprops=dict(color='black', linewidth=1.2),
                        flierprops=dict(marker='o', markersize=3,
                                        markeredgecolor='black', alpha=0.5))
        # color each box by session
        for patch, ses_id in zip(bp['boxes'], avail_ses):
            patch.set_facecolor(SES_COLORS[ses_id])
            patch.set_alpha(0.7)
    
    if sign_asterix:
        # add significance asterix for Mann-Whitney U test between ses01 and ses03
        stat, p = mannwhitneyu(box_data[0], box_data[1], alternative='two-sided')
        if p < 0.05:
            ax.text(np.mean(positions), 9, ha='center', va='center',
                    s='*', fontsize=FONT_SIZES['title'] + 8, color='k',)

    # ax.set_xlabel('Session', fontsize=FONT_SIZES['axes'])
    if not VIOLIN: ax.set_xticks(positions)
    else: ax.set_xticks([positions[0] - .15, positions[-1] + .15],)
    ax.set_xticklabels([SES_LABELS[ses_id] for ses_id in avail_ses],
                       fontsize=FONT_SIZES['axes'])
    ax.set_yticks(EMA_YTICKS)
    ax.set_yticklabels(EMA_YTICK_LABELS, fontsize=FONT_SIZES['ticks'],)
    ax.set_ylabel(f'Predicted {target_EMA_name} (EMA scale)', fontsize=FONT_SIZES['axes'])
    ax.tick_params(axis='both', size=FONT_SIZES['ticks'],
                   labelsize=FONT_SIZES['ticks'],)
    if return_ax:
        return ax
    
    plt.show()






def plot_daily_ft_mean(
    y_preds, y_times, session_code, ema_target_name,
    ax = None,
):
    # plot defaults
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        RETURN_AX = False
    else:
        RETURN_AX = True

    xtick_hop = 8


    avail_ses = np.unique(session_code)
    ses_preds = {ses_id: y_preds[session_code == ses_id]
                 for ses_id in avail_ses}
    ses_times = {ses_id: y_times[session_code == ses_id]
                 for ses_id in avail_ses}
    
    # plot each session separately
    for i_ses, ses_id in enumerate(avail_ses):
        # allocate all prediction based on their time within the day (in minutes)
        # returns day-pattern (mean and std) per block of default 15 min
        (
            ses_min_raster, ses_day_mean, ses_day_std
        ) = get_ft_daily_mean(ses_preds[ses_id], ses_times[ses_id], MINUTE_HOP=30,)
    
        ax.plot(ses_min_raster, ses_day_mean, lw=5, color=SES_COLORS[ses_id],
                alpha=.7, label=SES_LABELS[ses_id],)
        ax.fill_between(ses_min_raster, y1=ses_day_mean - ses_day_std,
                        y2=ses_day_mean + ses_day_std, alpha=.15,
                        color=SES_COLORS[ses_id],)

    ax.legend(fontsize=FONT_SIZES['title'], loc='upper right',)

    ax.set_xticks(ses_min_raster[::xtick_hop],)
    ax.set_xlim(ses_min_raster[0], ses_min_raster[-4])
    xtlabs = (np.array(ses_min_raster[::xtick_hop])/60).astype(int)
    ax.set_xticklabels([f'{h:02d}:00' for h in xtlabs], fontsize=FONT_SIZES['ticks'],)  
    ax.set_xlabel('Time at Day (hours)', fontsize=FONT_SIZES['axes'],)

    ax.set_ylim(1, 9)
    ax.set_yticks(EMA_YTICKS)
    ax.set_yticklabels(EMA_YTICK_LABELS, fontsize=FONT_SIZES['ticks'],)
    ax.set_ylabel(f'Predicted {ema_target_name} (EMA scale)', fontsize=FONT_SIZES['axes'],)

    ax.tick_params(labelsize=FONT_SIZES['ticks'], size=FONT_SIZES['ticks'], axis='both',)

    if RETURN_AX:
        return ax
    else:
        plt.show()


"""
back up, plot EMA daily mean, current data not sufficient

tempdf = ft_extr.get_feat_df_for_pred(
        sub_id=sub_id,
        ses_id=ses_id,
        ft_set_sel=FT_TYPE,
        FT_PARAMS_VERSION=FT_PARAMS_VERSION,
        ONLY_EMA_WINDOWS=True,
    )
    ema_values = tempdf[EMA_Y]
    ema_times = tempdf['timestamp']


    (
        ses_min_raster, ses_day_mean, ses_day_std
    ) = data_handling.get_ft_daily_mean(ema_values.values, ema_times, MINUTE_HOP=30,)

    ax.plot(ses_min_raster, ses_day_mean, lw=5, color=SES_COLORS[ses_id],
            alpha=.7,)
    ax.fill_between(ses_min_raster, y1=ses_day_mean - ses_day_std,
                    y2=ses_day_mean + ses_day_std, alpha=.15,
                    color=SES_COLORS[ses_id],)
"""

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
        f'Overall performance:\npearson-R: {round(pearson_r[0][0], 2)}, '
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


