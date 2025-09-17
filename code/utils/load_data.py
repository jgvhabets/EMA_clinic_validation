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
        
        # # SKIP ema31-32-33-34 as long as missing data not solved
        # if str(ema_id) in ['31', '32', '33', '34']:
        #     print(f'\n##### WARNING: HARDCODED EXCLUSING OF EMA32-33-34 bcs MISSINGs (get_ids())')
        #     continue 
        
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
    
    else:
        raise ValueError('EMA scales has to be 5 or 9')
    

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

    if np.isnan(score_og): score_new = np.nan
    else: score_new = convert_scores[score_og]


    return score_new


def get_EMA_UPDRS_data(
    condition='m0s0', SPLIT_TEST_TRAIN: str = 'none',
    CONVERT_SCORES: bool = True,
):
    """
    Load extracted EMA values from Excel file.
    
    IMPORTANT:
    From August 2025, EMA_val_scores.xlsx is double-checked
    by Aicha. Ema01 - ema17 are copied from Laura's table
    and contain inverted values for "Q3 - sadness" and
    "Q7 - tremor" (originally "Q6 - tremor" since
    "Q - Impulsivity" was not added yet, Q3 and Q7 for
    ema01-ema17 should be inverted to get the original.
    From ema18, Aicha's table contains the original EMA-
    values which were not converted by Laura's data collection.
    
    - converts 5-scale-likert-answers into 9-scale
    - converts directionality of answers, so "9" is
        always defined as best clinical answer
    """

    assert condition.lower() in ['m0s0', 'm0s1', 'm1s0', 'm1s1'], (
        'CONDITION SHOULD BE FORMAT MX-SX'
    )
    assert SPLIT_TEST_TRAIN.lower() in ['none', 'test', 'train'], (
        f'### Incorrect split-test-train: f{SPLIT_TEST_TRAIN}'
    )

    
    # load EMA and UPDRS data
    dat_folder = load_utils.get_onedrive_path('emaval_data')
    filepath = os.path.join(dat_folder, 'EMA_val_scores.xlsx')

    # load df with EMA and UPDRS tabs
    ema_df = read_excel(filepath, sheet_name='EMA')
    updrs_df = read_excel(filepath, sheet_name='UPDRS')
    
    # # get sub ids for EMA and LFP
    # ids = get_ids()

    
    ### UPDRS df selection
    # filter on defined condition (here, we lose all non-data rows!)
    updrs_df = updrs_df[updrs_df['condition'] == condition].reset_index(drop=True)


    ### EMA data, define which columns should be converted
    i_likert = np.where([v == 'score_likert' for v in ema_df['study_id']])[0][0]
    scale_convert_cols = ema_df.keys()[[v == 1 for v in ema_df.iloc[i_likert]]]
    if CONVERT_SCORES:
        i_dir_inv = np.where([v == 'directionality_inverse' for v in ema_df['study_id']])[0][0]  # define row with directionality inverse
        direct_convert_cols = ema_df.keys()[np.logical_and(
            [v == 1 for v in ema_df.iloc[i_likert]],
            [v == 1 for v in ema_df.iloc[i_dir_inv]]
        )]
    
    # filter on defined condition (here, we lose all non-data rows!)
    ema_df = ema_df[ema_df['condition'] == condition].reset_index(drop=True)


    ### Split test/training if defined
    if SPLIT_TEST_TRAIN.lower() == 'train':
        print(f'TODO: CREATE JSON WITH TEST STUDY IDS')
        EMA_SEL = [True] * ema_df.shape[0]
        UPDRS_SEL = [True] * updrs_df.shape[0]

    elif SPLIT_TEST_TRAIN.lower() == 'test':
        print(f'TODO: CREATE JSON WITH TEST STUDY IDS')
        EMA_SEL = [True] * ema_df.shape[0]
        UPDRS_SEL = [True] * updrs_df.shape[0]
    
    else:
        EMA_SEL = [True] * ema_df.shape[0]
        UPDRS_SEL = [True] * updrs_df.shape[0]

    # select data on defined selection
    ema_df = ema_df[EMA_SEL]
    updrs_df = updrs_df[UPDRS_SEL]


    ### convert all 5-scale answers to 9-scale
    # loop over columns and questions, select on likert scale aka need for conversion
    for (i_col, colname), i_row in product(
        enumerate(ema_df.keys()), np.arange(ema_df.shape[0])
    ):
        # skip non EMA item columns
        if not colname.startswith('Q'): continue
        # skip missing rows
        elif ema_df['missing'][i_row] == 1: continue
        # skip none data rows
        elif not ema_df['study_id'][i_row].startswith('ema'): continue
        # skip nans (due to varying EMA versions)
        elif np.isnan(ema_df[colname][i_row]): continue
        
        # convert 5 / 9-point scales of relevant scores
        if np.logical_and(ema_df['ema_scale'][i_row] == 5,
                        colname in scale_convert_cols):  # skip rows with correct scaling
            score_og = ema_df[colname][i_row]
            score_converted = ema_scale_converter(score_og, scale_og=5)
            ema_df.iloc[i_row, i_col] = score_converted
            # print(f'....row {i_row}, {colname}:\t{score_og} --> {score_converted} (5-9)')

        if CONVERT_SCORES:
            # convert EMA-directionality (9 always optimal clinical answer, and ASSUMES 9-SCALE)
            if np.logical_and(ema_df.iloc[i_dir_inv, i_col],
                              colname in direct_convert_cols):  # only for selected direct-changing questions
                score_og = ema_df[colname][i_row]
                score_converted = ema_directionality_converter(score_og)
                ema_df.iloc[i_row, i_col] = score_converted
                # print(f'....row {i_row}, {colname}:\t{score_og} --> {score_converted} (dir)')


        ### Re-invert ema01-17, Q3 (Sadness) and Q7 (Tremor)
        q_n = colname.split('_')[0]
        if q_n not in ['Q3', 'Q7']: continue
        id_n = ema_df['study_id'][i_row].split('ema')[1]
        if int(id_n) > 17: continue  # only re-invert ema01-17
        # re-invert
        score = ema_df[colname][i_row]
        score_REconverted = ema_directionality_converter(score)
        ema_df.iloc[i_row, i_col] = score_REconverted
        print(f'RE-INVERT... ema-N: {id_n}, {colname}:\t{score} --> {score_REconverted}')



    ### Sort df to emaID, due to unsorted excel table
    ema_df = ema_df.iloc[np.argsort(ema_df['study_id'])].reset_index(drop=True)
    updrs_df = updrs_df.iloc[np.argsort(updrs_df['study_id'])].reset_index(drop=True)


    return ema_df, updrs_df


def load_ema_df(sub_id, ses):

    # add home_dbs repo
    load_utils.add_home_repo()

    import load_raw.main_load_raw as load_home
    # import helper_functions.helpers as home_helpers
    import helper_functions.ema_utils as home_utils


    ses_class = load_home.load_subject(sub_id, ses)
    df = home_utils.extract_ema_reports(ses_class)
    df = home_utils.rename_questions(df, ses_class.EMA_reports_questions)

    df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)  # what is with index?
    
    return df


def filtered_submitted_ema_df(sub_id, ses):
    
    df = load_ema_df(sub_id, ses)
    df_sub = df[df['Submission'] == '1'].copy()
    df_sub['End Time'] = home_utils.convert_column_to_datetime(df_sub['End Time'])
    df_sub['Start Time'] = home_utils.convert_column_to_datetime(df_sub['Start Time'])
    df_sub['Scheduled Time'] = home_utils.convert_column_to_datetime(df_sub['Scheduled Time'])
    
    return df_sub