"""
Process different data modalities before analysis
"""
# custom
from utils.load_data import get_ids
from utils.load_utils import get_onedrive_path

# public
from os.path import dirname, join
import json
from pandas import DataFrame, isna
import numpy as np
from itertools import product
from mne.filter import filter_data



def get_subscores(df, dType, score_type='brady',):
    sel = {}
    # if data given is EMA
    if dType == 'EMA':
        sel['brady'] = ['Q6', 'Q10']  # 'movement, hands
        sel['gait'] = ['Q9',]  # gait
        sel['tremor'] = ['Q7',]  # tremor
        sel['nonmotor'] = ['Q1 ', 'Q2', 'Q3', 'Q4',]  # well being, motivation sadness energy
        # 'Q5' is impulsivity; 'Q8' is dyskinesia
    
    # is data is UPDRS
    elif dType == 'UPDRS':
        sel['brady'] = ['3', '4', '5', '6', '7', '8', '14']
        sel['gait'] = ['10', '11', '12']  # gehen, freezing, post-stab
        sel['tremor'] = ['15', '16', '17', '18',]  # tremor-rest, -post, -intent, -consist
        if score_type == 'nonmotor':
            raise ValueError('no nonmotor UPDRS-III subscores')
    
    col_sel = [any([k.startswith(x) for x in sel[score_type]])
               for k in df.keys()]
    
    return col_sel


def get_sum_df(EMA_dict, UPDRS_dict, MEAN_CORR: bool = True):

    ids = get_ids()

    SUMS = DataFrame(index=ids.index)

    for COND, datname, subscore in product(
        ['m0s0', 'm0s1', 'm1s0', 'm1s1'],
        ['EMA', 'UPDRS'],
        ['brady', 'tremor', 'gait', 'nonmotor']    
    ):
        # print(f'\nstart: {COND, datname, subscore}')

        # no nonmotor subscore in UPDRS
        if datname == 'UPDRS' and subscore == 'nonmotor':
            print('...skip nonmotor subscores for UPDRS')
            continue
        
        # get correct data dict
        if datname == 'EMA': DAT = EMA_dict
        elif datname == 'UPDRS': DAT = UPDRS_dict
        else: raise ValueError('datname must EMA or UPDRS')
        
        # select subscore items in resp data
        sel_bool = get_subscores(DAT[COND], score_type=subscore, dType=datname,)
        sel_cols = DAT[COND].keys()[sel_bool]
        sel_values = DAT[COND][sel_cols]
        
        # add mean value to new df
        # gives sumscore for sub-category per sub-id/cond-id
        SUMS[f'{datname}_SUM_{subscore}_{COND}'] = np.nansum(sel_values, axis=1)
        # # test max score for tremor
        # if subscore == 'tremor': SUMS[f'{datname}_SUM_{subscore}_{COND}'] = np.nanmax(sel_values, axis=1)
        
        # correct NaN for missing (before zeros)
        nan_sel = isna(sel_values).all(axis=1).values
        SUMS.loc[nan_sel, f'{datname}_SUM_{subscore}_{COND}'] = np.NaN   # * sum(nan_sel)


    # Correct sums with individual means
    if MEAN_CORR:
        # get individual mean over conditions
        for dtype, subscore in product(['EMA', 'UPDRS'],
                                       ['brady', 'tremor', 'gait', 'nonmotor']):
            # no nonmotor subscore in UPDRS
            if dtype == 'UPDRS' and subscore == 'nonmotor':
                continue

            sel = [k for k in SUMS.keys()
                   if k.startswith(f'{dtype}_SUM_{subscore}')]
            means = np.nanmean(SUMS[sel], axis=1)
            
            for COND in ['m0s0', 'm0s1', 'm1s0', 'm1s1']:
                # SUMS[f'{dtype}_SUM_{subscore}_{COND}'] = SUMS[f'{dtype}_SUM_{subscore}_{COND}'] - means
                SUMS[f'{dtype}_SUM_{subscore}_{COND}'] -= means
    
    return SUMS


def get_lfp_times():
    """Load JSON file with task timings during LFP recording"""
    dat_folder = get_onedrive_path('emaval')
    main_folder = dirname(dat_folder)
    filepath = join(main_folder, 'source_data', 'lfp_time_selections.json')

    # load timings
    with open(filepath, 'r') as f:
        times = json.load(f)

    return times


def lfp_filter(signal, Fs=250, low=2, high=48,):
    
    filtered = filter_data(
        data=signal,
        sfreq=Fs,
        l_freq=low,
        h_freq=high,
        method='fir',
        fir_window='hamming',
        verbose=False,
    )

    return filtered