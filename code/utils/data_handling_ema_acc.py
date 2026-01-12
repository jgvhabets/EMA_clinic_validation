
from dataclasses import dataclass, field
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os

from dbs_home.preprocessing import get_submovements


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