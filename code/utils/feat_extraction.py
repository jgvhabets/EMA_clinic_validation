

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import variation
import os
from pandas import read_csv, DataFrame
from itertools import compress

# current repo imports
from utils import acc_features as acc_fts
from utils import data_handling_ema_acc as data_handling
from utils.data_handling_ema_acc import windowData

# dbs_home imports
from dbs_home.utils import finding_paths
from dbs_home.preprocessing import acc_preprocessing as acc_prep
from dbs_home.preprocessing import get_submovements
import dbs_home.preprocessing.submovement_processing as submove_proc
import dbs_home.load_raw.load_watch_raw as load_watch



def get_features_per_session(
    sub_id,
    ses_id,
    home_dat,
    SUBMOVE_version = 'v3',
    ACC_SFREQ = 32,
    ACC_MIN_PER_EMA = 15,
    SM_MIN_DUR = .5,  # sec
    SM_MAX_DUR = 600,  # sec
    MIN_ACC_PRESENT = 0.5,
    LOAD_SAVE_FEATS = True,
    EXTRACT_FT_FROM_SMs = True,
    EXTRACT_FT_FULL_WIN = False,
    ACC_FEATS_on_SINGLE_MOVES = True,
    STORE_SUBMOVES = False,  # deprecated currently
    SAVE_PLOT = False,
    SHOW_PLOT = False,
):
    """
    """
    ### ft extraction params
    FEATS_INCL = [
        'rms_acc', 'svm_coefvar', 'svm_sd', 'iqr_acc',
        'lowpass_rms', 'jerk_magn', 'pow_4_7_ratio',
        'pow_8_12_ratio', 'pow_4_12', 'pow_1_3',
        'hr_mean', 'hr_std', 'hr_coefvar',
        'sm_count', 'sm_duration_mean',
        'sm_duration_std', 'sm_duration_coefvar'
    ]
    # EMA set variables
    EMA_CODING = {
        'tremor': 'Q7',
        'LID': 'Q8',
        'overall_move': 'Q6',
        'move_hands': 'Q10',
        'wellbeing': 'Q1'
    }

    # input checks
    assert not (EXTRACT_FT_FULL_WIN and EXTRACT_FT_FROM_SMs), (
        'CHOSE ONE OF TWO APROACHES, data OR times from submoves'
    )

    WIN_SAMPLES = (ACC_MIN_PER_EMA * 60 * ACC_SFREQ)

    # define storing paths
    FIGDIR = os.path.join(
        finding_paths.get_home_onedrive('figures'),
        'acc_processing', 'submovement_checks',
        f'submove_{SUBMOVE_version}', sub_id, ses_id
    )
    if not os.path.exists(FIGDIR): os.makedirs(FIGDIR)

    # TODO: at time of finalizing features, store exact feature extr settings
    FEATDIR = os.path.join(
        finding_paths.get_home_onedrive('data'),
        'features', 'ema_windows', sub_id
    )
    if not os.path.exists(FEATDIR): os.makedirs(FEATDIR)
    if EXTRACT_FT_FROM_SMs and ACC_FEATS_on_SINGLE_MOVES:
        fname_core = f'feats_singleSubmoves{SUBMOVE_version}'
    elif EXTRACT_FT_FROM_SMs:
        fname_core = f'feats_mergedSubmoves{SUBMOVE_version}'
    elif EXTRACT_FT_FULL_WIN:
        fname_core = 'feats_fullWin'
    else:
        raise ValueError('no valid FEAT EXTRACTION defined')
    feat_filename = f'{fname_core}_{sub_id}_{ses_id}.csv'


    ### LOAD FEATS if possible
    if LOAD_SAVE_FEATS:
        if feat_filename in os.listdir(FEATDIR):
            PRED_DF = read_csv(os.path.join(FEATDIR, feat_filename),
                                index_col=0,)

            return PRED_DF
    

    # if features are not directly extraced, list to store data
    if EXTRACT_FT_FROM_SMs:
        FEAT_STORE = {f: [] for f in FEATS_INCL}
        if ACC_FEATS_on_SINGLE_MOVES:  # replace ACC keys with _mean-cfvar keys
            for f in list(FEAT_STORE.keys()):
                if 'sm_' in f or 'hr_' in f: continue  # convert only svm-acc features
                for metric in ['SMmean', 'SMcfvar']:
                    FEAT_STORE[f'{f}_{metric}'] = []
                del FEAT_STORE[f]

    elif EXTRACT_FT_FULL_WIN:
        FEAT_STORE = {f: [] for f in FEATS_INCL if 'sm_' not in f}

    else:
        all_windows = []
        FEAT_STORE = None

    Y_STORE = {k: [] for k in list(EMA_CODING.keys())}
    TIME_STORE = []

    for i_day, str_day in enumerate(home_dat.watch_days):
        # define current day
        print(f"\n\n##### START day: {str_day}")
        # get dict for current day, needed in both methods of ft extraction (sm or not sm)
        try:
            day_dict_lists = acc_prep.get_day_EMA_AccWindows(
                subSesClass=home_dat, str_day=str_day,
            )
        except ValueError:
            print(f'DAY {str_day} failed: skipped')
            continue
        
        if not EXTRACT_FT_FROM_SMs:
            # no action required for EXTRACT_FT_FULL_WIN

            if STORE_SUBMOVES:
                (sm_day_starts,
                 sm_day_ends) = data_handling.get_submove_day_timestamps(
                    str_day, sub_id, ses_id,
                    SM_MIN_DUR=SM_MIN_DUR, SM_MAX_DUR=SM_MAX_DUR,
                    SUBMOVE_version=SUBMOVE_version,
                )
            
        else:
            # extract feats from sm-data directly
            sm_day_data = get_submovements.load_submovements(
                sub_id=sub_id, ses_id=ses_id, day=str_day,
                ONLY_TIMES=False,
                SUBMOVE_version=SUBMOVE_version,
            )
            if sm_day_data == None or type(sm_day_data) == type(None):
                print(f'DAY {str_day} has "None" as sm_day_data: skipped')
                continue
            # select submoves on durations
            sm_day_data = [s for s in sm_day_data if np.logical_and(
                s.duration > SM_MIN_DUR, s.duration < SM_MAX_DUR
            )]
            # get sm times for selection within window
            (sm_day_starts,
             sm_day_ends) = data_handling.get_submove_day_timestamps(
                str_day, sub_id, ses_id,
                SM_MIN_DUR=SM_MIN_DUR, SM_MAX_DUR=SM_MAX_DUR,
                SUBMOVE_version=SUBMOVE_version,
            )
        
        # load heartrate for full day
        hr_day_data = load_watch.get_source_heartrate_day(
            sub=sub_id, ses=ses_id, date=str_day,
        )


        for i_win in np.arange(len(list(day_dict_lists.values())[0])):

            print(f"\n\n\n######## START day-win # {i_win} / {len(list(day_dict_lists.values())[0])}")

            # create class with processed acc-data and with ema-dict per completed ema
            
            # skip incomplete acc data
            if len(day_dict_lists['acc_times'][i_win]) < (WIN_SAMPLES * MIN_ACC_PRESENT):
                print(f"skip WIN, not enough acc-data "
                    f"({len(day_dict_lists['acc_times'][i_win]) / (60 * ACC_SFREQ)} minutes)")
                continue

            # check for missing EMA data, and skip emas with missings
            if any(day_dict_lists['ema'][i_win].values == ''):
                print('skip WIN, missing EMA')
                continue
        
            # get window data
            windat = windowData(
                sub=sub_id,
                ses=ses_id,
                day=str_day,
                acc_times=day_dict_lists['acc_times'][i_win],
                acc_triax=day_dict_lists['acc_filt'][i_win],
                acc_svm=day_dict_lists['acc_svm'][i_win],
                ema=day_dict_lists['ema'][i_win],
            )

            # select heartrate for current window
            t1 = windat.acc_times[0]
            t2 = windat.acc_times[-1]
            hr_sel = np.logical_and(hr_day_data['timestamp'] > t1,
                                    hr_day_data['timestamp'] < t2)
            hr_win = hr_day_data[hr_sel].reset_index(drop=True)
            hr = [h if not h==0 else np.nan for h in hr_win[' HeartRate'].values]

            TIME_STORE.append(t1)

            # get ema window
            ema_win = day_dict_lists['ema'][i_win]

            # store window data for later ft-extraction
            if not EXTRACT_FT_FROM_SMs:

                if EXTRACT_FT_FULL_WIN:
                    # EXTRACT FEATURES from full acc-window, without submove-selection
                    win_ft_class = SubmoveData2Feat(
                        acc_svm=windat.acc_svm,
                        hr=hr_win[' HeartRate'].values,
                    )

                    # extract all feats that are defined in FEATS_INCL (no submove feats)
                    for ft in list(FEAT_STORE.keys()):
                        value = getattr(win_ft_class, f'run_{ft}')()  # extra brackets () for executing function
                        FEAT_STORE[ft].append(value)

                
                elif STORE_SUBMOVES:
                    # get mask for submove-pos samples in window
                    win_submove_bool = data_handling.get_window_submoveMask(
                        windat.acc_times, sm_day_starts, sm_day_ends,
                    )
                    # print(f'\nsubmove-positive window selection is '
                    #       f'{round(sum(win_submove_bool) / len(win_submove_bool) * 100)}%')
                    # change timeseries-attributes within window class
                    for att in ['acc_times', 'acc_svm', 'acc_triax']:
                        full_series = getattr(windat, att)
                        setattr(windat, att, full_series[win_submove_bool])
                    
                    # store durations of single selected submoves within window
                    sm_durations = sm_day_ends[win_submove_bool] - sm_day_starts[win_submove_bool]
                    setattr(windat, 'sm_durations', sm_durations)

                    all_windows.append(windat)

            # EXTRACT FEATURES from SubMovements directly, without substoring
            else:
                # EXTRACT FEATS FROM SUBMOVES DIRECTLY, no sub storing
                
                # get mask for submove-pos samples in window
                (win_submove_bool,
                submoves_in_win_bool) = data_handling.get_window_submoveMask(
                    windat.acc_times, sm_day_starts, sm_day_ends,
                )       

                # print(f'\nsubmove-positive window selection is '
                #       f'{round(sum(win_submove_bool)/len(win_submove_bool)*100)}%')

                # only submoves from day, are within current window
                sm_win_data = list(compress(sm_day_data, submoves_in_win_bool))

                # check correctness of submovements by plotting window ACC
                if SAVE_PLOT or SHOW_PLOT:
                    FIGNAME = f'submoveCheck_{SUBMOVE_version}_{windat.sub}_{windat.ses}_{str_day}_ema{i_win}'
                    data_handling.plot_submove_check(
                        FIGDIR=FIGDIR, FIGNAME=FIGNAME, SAVE_PLOT=SAVE_PLOT,
                        SHOW_PLOT=SHOW_PLOT, SUBMOVE_version=SUBMOVE_version,
                        windat=windat, win_submove_bool=win_submove_bool,
                        ema_win=ema_win, str_day=str_day,i_win=i_win, hr_win=hr_win,
                    )

                # EXTRACT FEATURES HERE FROM SUBMOVE data

                # get one array with svm data of all submovements from window            
                ### SM_MERGE
                merged_sm_svm = np.array([value for sm in sm_win_data for value in sm.svm])

                sm_ft_class = SubmoveData2Feat(
                    acc_svm=merged_sm_svm,
                    hr=hr_win[' HeartRate'].values,
                    sm_durations=np.array([s.duration for s in sm_win_data]),
                )

                # extract all feats that are defined in FEATS_INCL on MERGED SM data
                for ft in FEATS_INCL:
                    if np.logical_and(
                        ACC_FEATS_on_SINGLE_MOVES,
                        'sm_' not in ft and 'hr_' not in ft
                    ):
                        # do not add single merged acc feats if they be calculated on singlemoves
                        continue
                    value = getattr(sm_ft_class, f'run_{ft}')()  # extra brackets () for executing function
                    FEAT_STORE[ft].append(value)

                ### SM SINGLES
                if ACC_FEATS_on_SINGLE_MOVES:
                    for ft in [f for f in FEATS_INCL if ('sm_' not in f and 'hr_' not in f)]:
                        # skip heartrate and sm-duration features here
                        ft_win_list = []  # store ft-values per sm within window
                        if len(sm_win_data) == 0:
                            ft_win_list.append(0)
                        else:
                            for sm in sm_win_data:
                                single_sm_class = SubmoveData2Feat(acc_svm=sm.svm,)
                                value = getattr(single_sm_class, f'run_{ft}')()
                                ft_win_list.append(value)  # add value per submovement
                        # add summarized scores per window
                        FEAT_STORE[f'{ft}_SMmean'].append(np.nanmean(ft_win_list))
                        FEAT_STORE[f'{ft}_SMcfvar'].append(variation([v for v in ft_win_list if not np.isnan(v)]))


                # TODO: try for cnn -> interpolate single sm-features into 
                # n=100 (zB) mask, 0-padding for n-sm <100 windows
            
            ### extract EMA values per window
            for EMA_ITEM in list(Y_STORE.keys()):
                Y_STORE[EMA_ITEM].append(float(ema_win[EMA_CODING[EMA_ITEM]]))
                
    if LOAD_SAVE_FEATS:
        # get one df with all info
        PRED_DF = DataFrame(TIME_STORE, columns=['timestamp'])
        for k, v in FEAT_STORE.items():
            PRED_DF[k] = v
        for k, v in Y_STORE.items():
            PRED_DF[k] = v
        PRED_DF.to_csv(os.path.join(FEATDIR, feat_filename))





@dataclass
class SubmoveData2Feat:
    """
    - svm: array of merged svm-arrays of i.e. all submovements
        within a window
    """
    acc_svm: np.ndarray = field(default_factory=lambda: np.array([]))
    sm_durations: np.ndarray = field(default_factory=lambda: np.array([]))
    hr: np.ndarray = field(default_factory=lambda: np.array([]))
    sfreq: int = 32

    # functions standby, executed on active calling
    
    ### SUBMOVEMENT FEATURES

    def run_sm_count(self):
        value = len(self.sm_durations)
        return value
    
    def run_sm_duration_mean(self):
        value = np.nanmean(self.sm_durations)
        if np.isnan(value): value = 0
        return value
    
    def run_sm_duration_std(self):
        value = np.nanstd(self.sm_durations)
        if np.isnan(value): value = 0
        return value
    
    def run_sm_duration_coefvar(self):
        value = variation(self.sm_durations)
        if np.isnan(value): value = 0
        return value
    

    ### ACC FEATURES

    def run_rms_acc(self):
        value = acc_fts.get_rms_acc(self)
        # normalize to length of submovement data
        value = value / len(self.acc_svm)
        return value
    
    def run_svm_coefvar(self):
        value = acc_fts.get_svm_var(self)
        if np.isnan(value): value = 0
        return value
    
    def run_svm_sd(self):
        value = acc_fts.get_svm_sd(self)
        return value
    
    def run_iqr_acc(self):
        value = acc_fts.get_iqr_acc(self)
        return value
    
    def run_lowpass_rms(self):
        value = acc_fts.get_lp_rms_acc(self)
        # normalize to length of submovement data
        value = value / len(self.acc_svm)
        return value
    
    def run_jerk_magn(self):
        value = acc_fts.get_jerk_magn(self=self)
        return value
          
    def run_pow_4_7_ratio(self):
        value = acc_fts.get_pow(self, f_lo=4, f_hi=7,
                                include_low=True, include_high=True,
                                RATIO_TO_ALL_HZ=True,)
        if np.isnan(value): value = 0
        return value
    
    def run_pow_8_12_ratio(self):
        value = acc_fts.get_pow(self, f_lo=8, f_hi=12,
                                include_low=True, include_high=True,
                                RATIO_TO_ALL_HZ=True,)
        if np.isnan(value): value = 0
        return value
    
    def run_pow_4_12(self):
        value = acc_fts.get_pow(self, f_lo=4, f_hi=12,
                                include_low=True, include_high=True,
                                RATIO_TO_ALL_HZ=False,)
        return value
    
    def run_pow_1_3(self):
        value = acc_fts.get_pow(self, f_lo=1, f_hi=3,
                                include_low=True, include_high=True,
                                RATIO_TO_ALL_HZ=False,)
        return value

    
    # HEARTRATE FEATURES
    def run_hr_mean(self):
        value = np.nanmean(self.hr)
        return value
    
    def run_hr_std(self):
        value = np.nanstd(self.hr)
        return value

    def run_hr_coefvar(self):
        value = variation(self.hr)
        return value


    # initiated on creation
    def __post_init__(self):
        # if window doesnt have any submovements, set 0-array to generate zeros and NaNs
        if len(self.acc_svm) == 0:
            setattr(self, 'acc_svm', np.zeros(self.sfreq))

        # always set psds ready for other features
        self.fx, self.psx = acc_fts.get_psd(self)

        # clean heartrate up, replace 0s with NaNs
        if len(self.hr) > 0:
            # self.hr_clean = [h if not h==0 else np.nan for h in self.hr]
            self.hr = [v if v != 0 else np.nan for v in self.hr]


