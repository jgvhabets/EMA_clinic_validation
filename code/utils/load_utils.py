import sys
from os import listdir, getcwd
from os.path import join, dirname, abspath, exists
import json
from datetime import date
from datetime import datetime
from itertools import compress
from numpy import array, sqrt, nanmean, logical_and


def select_period(day_list: list,
                  period_firstday: str,
                  period_lastday: str):
    
    excl_days = []

    for day in day_list:
        if isinstance(period_firstday, str):
            if date.fromisoformat(day) < date.fromisoformat(period_firstday):
                excl_days.append(day)
        if isinstance(period_lastday, str):  
            if date.fromisoformat(day) > date.fromisoformat(period_lastday):
                excl_days.append(day)
    
    if len(excl_days) > 0:
        print(f'exclude days: {excl_days}')
        sel_days = [d not in excl_days for d in day_list]
        day_list = list(compress(day_list, sel_days))
    
    return day_list




def add_PyPerceive_repo():
    # change enter path as follows: os.path.abspath(r'X:\xxxx\xxxx\PyPerceive/code')
    # keep exact formatting: pp_code_path = os.path.abspath(r'PATH')
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    pp_code_path = fr'{paths["pyperceive_path"]}'
    # print(f'pyPerceive-path extracting from JSON: {pp_code_path}')

    # find path if not given
    if (isinstance(pp_code_path, str) and
        exists(abspath(pp_code_path)) == False):
        
        path = getcwd()
        for i in range(20):
            if 'PyPerceive' in listdir(path):
                pp_code_path = join(path, 'PyPerceive', 'code')
                break
            else:
                path = dirname(path)

    sys.path.append(pp_code_path)
    print(f'PyPerceive folder added: {pp_code_path}')


def get_onedrive_path(folder: str = 'emaval', USER='jeroen',):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    """
    folder_options = ['onedrive', 'home', 'figures', 'data', 'emaval',
                      'emaval_fig', 'emaval_data', 'emaval_dat']
    
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options} NOCAPITALS')

    path = getcwd()

    while_count = 0
    while dirname(path)[-5:].lower() != 'users':
        path = dirname(path)
        while_count += 1
        if while_count > 20: return False

    # path is now Users/username
    onedrive_f = [f for f in listdir(path) if logical_and(
        'onedrive' in f.lower(), 'charit' in f.lower()
    )]
    path = join(path, onedrive_f[0])  # onedrive path
    homepath = join(path, 'HOME monitoring PREP')
    emaval_path = join(homepath, 'PROJECTS', 'EMA validation')

    if folder.lower() == 'onedrive': return path

    elif folder.lower() == 'home':
        return homepath
    
    elif 'emaval' in folder.lower():
    
        if 'fig' in folder.lower():
            return join(emaval_path, 'figures')
        elif 'dat' in folder.lower():
            return join(emaval_path, 'data', 'EMA_UPDRS_DATA')
        else:
            return emaval_path
    
    elif folder.lower() == 'data': return join(emaval_path, 'data', 'EMA_UPDRS_DATA')

    elif folder.lower() == 'figures': return join(emaval_path, 'figures')

    else: return emaval_path

