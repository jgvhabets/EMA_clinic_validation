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
