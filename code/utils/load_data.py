"""
Import neural local field potential
recordings
"""

import numpy as np
import os
import json
from datetime import datetime as dt
from pandas import read_excel, DataFrame, isna
from dataclasses import dataclass
from typing import Any
from itertools import compress

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
    dat_folder = load_utils.get_onedrive_path('emaval')
    main_folder = os.path.dirname(dat_folder)
    sub_df = read_excel(os.path.join(main_folder, 'ema_percept_id_overview.xlsx'))

    # get empty df for IDs
    ids = DataFrame(columns=['ema_id', 'prc_id', 'prc_ses'])

    for i, ema_id in enumerate(sub_df['ema_id']):
        # extract IDs and add "0"s to IDs (7 -> 07)
        ema_id = "{:02d}".format(ema_id)
        prc_id = "{:03d}".format(sub_df.iloc[i]['percept_id'])
        prc_ses = sub_df.iloc[i]['session']
        
        ids.loc[f'ema{ema_id}'] = [ema_id, prc_id, prc_ses]


    return ids


def get_EMA_UPDRS_data(condition='m0s0',):
    """Load extracted EMA values from Excel file"""

    assert condition.lower() in ['m0s0', 'm0s1', 'm1s0', 'm1s1'], (
        'CONDITION SHOULD BE FORMAT MX-SX'
    )
    dat_folder = load_utils.get_onedrive_path('emaval')
    main_folder = os.path.dirname(dat_folder)
    # rename condition
    condition = condition.replace('m', 'med')
    condition = condition.replace('s', '-stim')
    condition = condition.replace('0', 'OFF')
    condition = condition.replace('1', 'ON')
    # load EMA and UPDRS data
    df = read_excel(
        os.path.join(main_folder, 'EMA_UPDRS_recording_data.xlsx'),
        sheet_name=condition
    )
    # get sub ids for EMA and LFP
    ids = get_ids()

    # get UPDRS relevant columns
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

    # get EMA relevant columns
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


    return ema_df, updrs_df


