"""
Process different data modalities before analysis
"""
# custom
from utils.load_data import get_ids
from utils.load_utils import get_onedrive_path

# public
from os.path import dirname, join
import json
import pandas as pd
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

    SUMS = pd.DataFrame(index=ids.index)

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
        sum_values = np.nansum(sel_values, axis=1)

        if datname == 'EMA' and subscore == 'brady':
            sum_values /= 2  # EMA brady exists of two questions, take mean answer
        elif datname == 'EMA' and subscore == 'nonmotor':
            sum_values /= 4  # EMA nonmotor exists of four questions, take mean answer
        
        SUMS[f'{datname}_SUM_{subscore}_{COND}'] = sum_values
        # # test max score for tremor
        # if subscore == 'tremor': SUMS[f'{datname}_SUM_{subscore}_{COND}'] = np.nanmax(sel_values, axis=1)
        
        # correct NaN for missing (before zeros)
        nan_sel = pd.isna(sel_values).all(axis=1).values
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
    
    # Remove missings or corrupted data
    SUMS = remove_missing_and_corrupts(SUMS)

    return SUMS


def remove_missing_and_corrupts(sumdf):

    conds = ['m0s0', 'm0s1', 'm1s0', 'm1s1']
    LIST_CORRUPTED = ['ema22', 'ema24']  # corrupted due to EMA app error

    list_all_miss = []

    for sub in sumdf.index:

        allmiss = all([np.isnan(sumdf.loc[sub, f'UPDRS_SUM_brady_{c}'])
                        for c in conds])
        if allmiss: list_all_miss.append(sub)

        allmiss = all([np.isnan(sumdf.loc[sub, f'EMA_SUM_brady_{c}'])
                        for c in conds])
        if allmiss: list_all_miss.append(sub)

    ### EMA of ema22 and ema24 are corrupted
    list_all_miss.extend(LIST_CORRUPTED)

    sumdf = sumdf.drop(index=list_all_miss)

    return sumdf


def get_lmm_df(sumdf):

    conditions = ['m0s0', 'm0s1', 'm1s0', 'm1s1']

    # craete lmm df with all info in same columns
    lmm_df = pd.DataFrame(
        columns=[
            'subid', 'cond'] + [
            c.split('_m0s0')[0] for c in sumdf.columns if 'm0s0' in c
        ],
        index=[f'{s}_{c}' for s, c in product(sumdf.index, conditions)],
    )
    # fill columns from sumdf
    for i, idx in enumerate(lmm_df.index):
        sub, con = idx.split('_')
        lmm_df.loc[idx, 'subid'] = int(sub.split('ema')[1])  # takes ema id number as int
        lmm_df.loc[idx, 'cond'] = np.where(np.array(conditions) == con)[0][0]  # decodes in order of conditions

        for i_col, col in enumerate(lmm_df.columns):
            if col in ['subid', 'cond']: continue
            lmm_df.loc[idx, col] = sumdf.loc[sub, f'{col}_{con}']

    # remove nan rows
    drop_idx = [i for i in lmm_df.index if any(pd.isna(lmm_df.loc[i]))]
    # print(drop_idx)
    lmm_df = lmm_df.drop(index=drop_idx).astype(np.float64)  # ensure floats for lmm model


    return lmm_df




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


def get_train_test_split(sumdf):
    
    states = ['m0s0', 'm0s1', 'm1s0', 'm1s1']

    sub_nans = {}

    shf_ids = list(sumdf.index).copy()
    np.random.seed(27)
    np.random.shuffle(shf_ids)

    for subid in shf_ids:
        ema_nan = sum(
            [np.isnan(sumdf.loc[subid, f'EMA_SUM_brady_{state}'])
            for state in states]
        )
        updrs_nan = sum(
            [np.isnan(sumdf.loc[subid, f'UPDRS_SUM_brady_{state}'])
            for state in states]
        )
        sub_nans[subid] = max(ema_nan, updrs_nan)

    n_no_nans = sum([v == 0 for v in sub_nans.values()])
    n_nans = len(sub_nans.values()) - n_no_nans
    mean_nan = np.mean([v for v in sub_nans.values() if v > 0])

    print(f'no-NaN: {n_no_nans}, n-Nans: {n_nans} (mean NaNs / sub = {mean_nan})')


    # SPLIT
    fullsubs = [s for s, m in sub_nans.items() if m == 0]
    nansubs = [s for s, m in sub_nans.items() if m > 0]

    test_subs = fullsubs[:4] + nansubs[:4]
    train_subs = fullsubs[4:] + nansubs[4:]

    return train_subs, test_subs