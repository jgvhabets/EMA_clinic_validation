"""
Import neural local field potential
recordings
"""

import sys
import numpy as np
import os
import json
from datetime import datetime as dt
from pandas import read_excel, DataFrame, isna
from dataclasses import dataclass
from typing import Any
from itertools import compress, product

import utils.load_utils as load_utils

try:
    from PerceiveImport.classes import main_class
except:
    # if PyPerceive import does not work
    from utils.load_utils import add_PyPerceive_repo
    # first add repo directory and try again
    add_PyPerceive_repo()
    from PerceiveImport.classes import main_class


def add_PyPerceive_repo():
    # change enter path as follows: os.path.abspath(r'X:\xxxx\xxxx\PyPerceive/code')
    # keep exact formatting: pp_code_path = os.path.abspath(r'PATH')
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    pp_code_path = fr'{paths["pyperceive_path"]}'
    print(f'path extracting from JSON: {pp_code_path}')

    # find path if not given
    if (isinstance(pp_code_path, str) and
        os.path.exists(os.path.abspath(pp_code_path)) == False):
        
        path = os.getcwd()
        for i in range(20):
            if 'PyPerceive' in os.listdir(path):
                pp_code_path = os.path.join(path, 'PyPerceive', 'code')
                break
            else:
                path = os.path.dirname(path)

    sys.path.append(pp_code_path)
    print(f'PyPerceive folder added: {pp_code_path}')


def get_percept_code(self):
    """
    Converts EMA-study code into Percept-code, works
    within loadSubject-class
    """

    with open(os.path.join(self.paths["sourcedata_path"],
                           "rec_info.json"), 'r') as f:
        rec_info = json.load(f)
        self.sub_lfp = rec_info["prc_codes"][self.sub]

    print(f'study sub {self.sub} linked to percept sub-{self.sub_lfp}')


def get_ids():
    """Load Excel file with subject IDs"""
    dat_folder = load_utils.get_onedrive_path('emaval_data')
    sub_df = read_excel(os.path.join(dat_folder, 'ema_percept_id_overview.xlsx'))

    # get empty df for IDs
    ids = DataFrame(columns=['ema_id', 'prc_id', 'prc_ses'])

    for i, ema_id in enumerate(sub_df['ema_id']):
        # extract IDs and add "0"s to IDs (7 -> 07)
        ema_id = "{:02d}".format(ema_id)
        prc_id = "{:03d}".format(sub_df.iloc[i]['percept_id'])
        prc_ses = sub_df.iloc[i]['session']
        
        ids.loc[f'ema{ema_id}'] = [ema_id, prc_id, prc_ses]


    return ids


def ema_scale_converter(score_og, scale_og=5,):

    if scale_og == 5:

        convert_scores = {
            1: 1,
            2: 3,
            3: 5,
            4: 7,
            5: 9
        }
    
    elif scale_og == 9:

        convert_scores = {
            1: 1,
            2: 1.5,
            3: 2,
            4: 2.5,
            5: 3,
            6: 3.5,
            7: 4,
            8: 4.5,
            9: 5
        }
    
    score_new = convert_scores[score_og]

    return score_new


def ema_directionality_converter(score_og,):

    convert_scores = {
        1: 9,
        2: 8,
        3: 7,
        4: 6,
        5: 5,
        6: 4,
        7: 3,
        8: 2,
        9: 1
    }
    
    score_new = convert_scores[score_og]

    return score_new


def get_EMA_UPDRS_data(condition='m0s0',):
    """
    Load extracted EMA values from Excel file
    
    - converts 5-scale-likert-answers into 9-scale
    - converts directionality of answers, so "9" is
        always defined as best clinical answer
    """

    assert condition.lower() in ['m0s0', 'm0s1', 'm1s0', 'm1s1'], (
        'CONDITION SHOULD BE FORMAT MX-SX'
    )
    dat_folder = load_utils.get_onedrive_path('emaval_data')

    # rename condition
    condition2 = condition.replace('m', 'med')
    condition2 = condition2.replace('s', '-stim')
    condition2 = condition2.replace('0', 'OFF')
    condition2 = condition2.replace('1', 'ON')

    # load EMA and UPDRS data  -> currently only for UPDRS
    filepath = os.path.join(dat_folder, 'EMA_UPDRS_recording_data.xlsx')
    # print(f'{filepath}\n\texists?  -> {os.path.exists(filepath)}')
    df = read_excel(filepath, sheet_name=condition2)  # deprecated EMA workflow
    # get sub ids for EMA and LFP
    ids = get_ids()

    ### get UPDRS relevant columns
    col_ns = np.where([str(x).startswith('3') for x in df.iloc[0]])[0]
    col_bool = [str(x).startswith('3') for x in df.iloc[0]]
    col_names = df.iloc[0][col_bool].values
    # create empty df with UPDRS columns
    updrs_df = DataFrame(columns=col_names)
    # fill rows with UPDRS sub data
    for i, s in enumerate(df['study_code']):
        if isinstance(s, str):
            row = np.where(df['study_code'] == s)[0][0]
            updrs_df.loc[s] = df.iloc[row][col_bool].values


    ### get EMA data
    ema_file = os.path.join(dat_folder, 'EMA_val_scores.xlsx')
    ema_df = read_excel(ema_file, sheet_name='EMA')
    # define which columns should be converted
    i_dir_inv = np.where([v == 'directionality_inverse' for v in ema_df['study_id']])[0][0]
    i_likert = np.where([v == 'score_likert' for v in ema_df['study_id']])[0][0]
    scale_convert_cols = ema_df.keys()[[v == 1 for v in ema_df.iloc[i_likert]]]
    direct_convert_cols = ema_df.keys()[np.logical_and(
        [v == 1 for v in ema_df.iloc[i_likert]],
        [v == 1 for v in ema_df.iloc[i_dir_inv]]
    )]
    
    # filter on defined condition (here, we lose all non-data rows!)
    ema_df = ema_df[ema_df['condition'] == condition].reset_index(drop=True)

    # define row with directionality inverse

    ### convert all 5-scale answers to 9-scale
    # loop over columns and questions, select on likert scale aka need for conversion
    for (i_col, colname), i_row in product(
        enumerate(ema_df.keys()), np.arange(ema_df.shape[0])
    ):
        # skip non EMA item columns
        if not colname.startswith('Q'): continue
        # skip missing rows
        elif ema_df['missing'][i_row]: continue
        # skip none data rows
        elif not ema_df['study_id'][i_row].startswith('ema'): continue
        # skip nans (due to varying EMA versions)
        elif np.isnan(ema_df[colname][i_row]): continue
        
        # convert 5 / 90-point scales of relevant scores
        if np.logical_and(ema_df['ema_scale'][i_row] == 5,
                          colname in scale_convert_cols):  # skip rows with correct scaling
            score_og = ema_df[colname][i_row]
            score_converted = ema_scale_converter(score_og, scale_og=5)
            ema_df.iloc[i_row, i_col] = score_converted

        # convert directionality (9 is always best clinical answer)
        # ASSUMES 9-SCALE ANSWER VALUE
        if np.logical_and(ema_df.iloc[i_dir_inv, i_col],
                          colname in direct_convert_cols):  # only for selected direct-changing questions
            score_og = ema_df[colname][i_row]
            score_converted = ema_directionality_converter(score_og)
            ema_df.iloc[i_row, i_col] = score_converted




    # deprecated data structure
    """
        filepath = os.path.join(dat_folder, 'EMA_UPDRS_recording_data.xlsx')
        # print(f'{filepath}\n\texists?  -> {os.path.exists(filepath)}')
        df = read_excel(filepath, sheet_name=condition)  # deprecated EMA workflow
        
        col_ns = np.where([str(x).startswith('Q') for x in df.iloc[0]])[0]
        col_bool = [str(x).startswith('Q') for x in df.iloc[0]]
        col_names = df.iloc[0][col_bool].values
        # create empty df with UPDRS columns
        ema_df = DataFrame(columns=col_names)
        # fill rows with UPDRS sub data
        for i, s in enumerate(df['study_code']):
            if isinstance(s, str):
                row = np.where(df['study_code'] == s)[0][0]
                ema_df.loc[s] = df.iloc[row][col_bool].values
    """

    return ema_df, updrs_df


