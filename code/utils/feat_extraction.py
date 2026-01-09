

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import variation


from utils import acc_features as acc_fts



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


