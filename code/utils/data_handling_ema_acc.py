
from dataclasses import dataclass, field
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from pandas import read_excel, isna

from dbs_home.preprocessing import get_submovements
from dbs_home.utils.finding_paths import get_home_onedrive

@dataclass(init=True,)
class windowData:
    sub: str
    ses: str
    day: str | None = None
    acc_times: np.ndarray | None = None
    acc_triax: np.ndarray | None = None
    acc_svm: np.ndarray | None = None
    sfreq: int | None = None
    ema: dict = field(default_factory=dict)
    day: str | None = None

    def __post_init__(self):

        print(f'created windowData class for {self.sub}, {self.ses};'
              f'starttime {self.acc_times[0]}')
        if type(self.day) == str: print(f'belonging to day {self.day}')

        if self.sfreq == None:
            # extract sfreq if not given
            time_df = np.diff(self.acc_times[:5])[0]
            self.sfreq = int(dt.timedelta(seconds=1) / time_df)

        

def get_submove_day_timestamps(
    day, sub, ses, SM_MIN_DUR=0., SM_MAX_DUR=60.,
    SUBMOVE_version='v1',
):

    sm_day_times = get_submovements.load_submovements(
        sub_id=sub, ses_id=ses, day=day,
        ONLY_TIMES=True, SUBMOVE_version=SUBMOVE_version,
    )

    # get submovement start and ends (from json-dict)
    sm_time_arr = np.array([list(s.values()) for s in sm_day_times['submovements']])
    # get array with datetime objects for starts and ends
    sm_day_starts = np.array([dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
                            for t in sm_time_arr[:, 0]])
    sm_day_ends = np.array([dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
                        for t in sm_time_arr[:, 1]])
    
    sm_durations = sm_day_ends - sm_day_starts
    sel_submoves = np.logical_and(
        sm_durations > dt.timedelta(seconds=SM_MIN_DUR),
        sm_durations < dt.timedelta(seconds=SM_MAX_DUR),
    )
    sm_day_starts = sm_day_starts[sel_submoves]
    sm_day_ends = sm_day_ends[sel_submoves]
    
    return sm_day_starts, sm_day_ends


def get_window_submoveMask(win_times, sm_dt_starts, sm_dt_ends,):
    """
    get boolean array that is "1" for submovement-positive samples
    during EMA-window-matched acc data

    returns:
    - array with shape acc-window, positive for submove samples
    - boolean array for day-submoves that are within current window-times
    """

    # get start and end time of acc-ema window
    win_start, win_end = win_times[0], win_times[-1]

    # compare and select starts and ends within acc-ema-window
    submoves_in_win_mask = np.logical_and(
        sm_dt_starts > win_start,
        sm_dt_ends < win_end
    )
    win_sm_starts = sm_dt_starts[submoves_in_win_mask]
    win_sm_ends = sm_dt_ends[submoves_in_win_mask]

    # select window-samples that are within submoves
    # create boolean for acc-window, that will be 1 during submoves
    win_submove_bool = np.zeros_like(win_times)
    for t1, t2 in zip(win_sm_starts, win_sm_ends):
        mask = np.logical_and(win_times > t1, win_times < t2)
        win_submove_bool[mask] = 1

    win_submove_bool = win_submove_bool.astype(bool)

    return win_submove_bool, submoves_in_win_mask



def plot_submove_check(
    FIGDIR, FIGNAME, SAVE_PLOT, SHOW_PLOT,
    windat, win_submove_bool, ema_win, hr_win,
    str_day, i_win, SUBMOVE_version,
):

    FONTSIZE = 12
    
    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    ax2.plot(hr_win['timestamp'], hr_win[' HeartRate'], color='orangered',)
    ax2.set_ylim(-10, 130)
    ax2.set_ylabel('Heartrate (bpm)', size=FONTSIZE, color='orangered')

    ax.plot(windat.acc_times, windat.acc_svm, label='svm', alpha=.5,)
    ax.scatter(windat.acc_times, win_submove_bool.astype(int),
                label='submove-boolean', s=50, alpha=.3, color='orange',)
    ax.set_ylim(-.5, 5)
    ax.set_ylabel('ACC-vector (squared-magn.)',
                    size=FONTSIZE, color='blue')
    ax.legend(loc='upper right')
    ax.set_title(
        f'{str_day}: EMA-window # {i_win} ({windat.sub}, {windat.ses}, submove-{SUBMOVE_version})'
        f'\n EMA: tremor: {ema_win["Q7"]}, dyskinesia: {ema_win["Q8"]}'
    )

    for axx in [ax, ax2]:
        axx.tick_params(axis='both', size=FONTSIZE,
                        labelsize=FONTSIZE,)
    plt.tight_layout()

    if SAVE_PLOT:
        plt.savefig(os.path.join(FIGDIR, FIGNAME), facecolor='w', dpi=150,)

    if SHOW_PLOT: plt.show()
    else: plt.close()



def get_med_scheme(sub, ses):

    # get med-scheme
    f_path = os.path.join(get_home_onedrive('raw_data'),
                        sub, f'med_schemes_{sub}.xlsx')

    med_scheme = read_excel(f_path, sheet_name=ses, index_col='intake_time')

    return med_scheme


def get_intervals_to_ldopa(timestamps, sub, ses):
    """
    returned as minutes, positive if time was after ldopa intake,
    negative if time was prior to ldopa intake
    """

    med_scheme = get_med_scheme(sub, ses)
    # select l-dopa
    ldopa_scheme = med_scheme['levodopa'][~isna(med_scheme['levodopa'])]
    ldopa_times = ldopa_scheme.index.values
    ldopa_clocktimes = [dt.datetime.strptime(f'{t.hour}:{t.minute}', '%H:%M')
                        for t in ldopa_times]
    # print(ldopa_clocktimes)
    sample_clocktimes = [dt.datetime.strptime(f'{t.hour}:{t.minute}', '%H:%M')
                         for t in timestamps]

    # calculate differences in clock time
    intervals = []

    for i_t, t in enumerate(sample_clocktimes):  # i for debugging
        # print('\n', t, timestamps[i_t])
        i_closest_dopa = np.argmin(np.abs(t - np.array(ldopa_clocktimes)))
        closest_dopa_time = ldopa_clocktimes[i_closest_dopa]
        dopa_distance = t - closest_dopa_time
        # print(dopa_distance)
        intervals.append(dopa_distance)  # is positive if time is after closest intake, neg if t is pre-intake

    # convert timedeltas into minutes
    interval_minutes = []
    for t in intervals:
        if t.days == 0:
            interval_minutes.append(t.seconds / 60)
        elif t.days == -1:
            interval_minutes.append(t.seconds/60 - (24*60) )

    return interval_minutes



def sort_values_into_ldopa_distances(values, dopa_distances,):

    dist_borders = np.arange(-90, 91, step=15,)

    border_groups = [[] for b in dist_borders]

    for t_dist, t_val in zip(dopa_distances, values):

        if t_dist <= dist_borders[0]:
            border_groups[0].append(t_val)
        elif t_dist >= dist_borders[-1]:
            border_groups[-1].append(t_val)
        else:
            i_group = np.argmin(np.abs(dist_borders - t_dist))
            border_groups[i_group].append(t_val)
            # print(f'{t_dist} added to {dist_borders[i_group]}')
    
    return border_groups, dist_borders


def get_dayminutes(t):

    if type(t) == str:
        t = dt.datetime.strptime(t[:16], "%Y-%m-%d %H:%M")

    minutes = t.minute
    minutes += (t.hour * 60)

    return minutes


def get_daily_minutes_mask(HR_START=8, HR_END=22, WIN_LEN_minutes=15):

    # get day time raster (in minutes into day)
    t0 = dt.datetime.strptime(str(HR_START), "%H")
    n_wins = (HR_END - HR_START) * (60 / WIN_LEN_minutes)
    mask_dtimes = [t0 + dt.timedelta(minutes=int(WIN_LEN_minutes * i))
                   for i in np.arange(n_wins)]
    mask_minutes = [get_dayminutes(t) for t in mask_dtimes]

    return mask_minutes


def get_ft_daily_mean(
    ft_values, ft_times,
    PLOT_SAMPLE_DISTRIBUTION=False,
):
    
    mask_minutes = get_daily_minutes_mask()
    mask_dict = {m: [] for m in mask_minutes}

    for i, v in enumerate(ft_values):
        # get daily minute of timestamp (df-index)
        t = get_dayminutes(ft_times[i])
        # add ft value to list of corresponding daily minute
        try:
            mask_dict[t].append(v)
        except KeyError:
            # daily minutes not correct, find closest
            i_alt = np.argmin(abs(np.array(mask_minutes) - t))
            min_alt = mask_minutes[i_alt]
            mask_dict[min_alt].append(v)
    
    daily_minutes = list(mask_dict.keys())
    daily_mean = np.array([np.nanmean(l) for l in mask_dict.values()])
    daily_std = np.array([np.nanstd(l) for l in mask_dict.values()])

    ### plot distribution of samples on daily-minutes-raster
    if PLOT_SAMPLE_DISTRIBUTION:
        for k, v in mask_dict.items():
            print(f'{k} min, {k / 60} h: {len(v)} samples')

        samples_at_minutes = [len(l) for l in mask_dict.values()]
        print(samples_at_minutes)
        fig,ax = plt.subplots(1,1, figsize=(8, 3))
        ax.bar(x=daily_minutes, height=samples_at_minutes,
               width=12, color='olivedrab', alpha=.3,)
        # ax.plot(daily_minutes, [len(l) for l in mask_dict.values()])
        ax.set_ylabel('n samples (count)')
        ax.set_xlabel('Time at Day (hours)')
        ax.set_xticks(daily_minutes[::8])
        ax.set_xticklabels((np.array(daily_minutes[::8])/60).astype(int),)

        plt.show()

    return daily_minutes, daily_mean, daily_std