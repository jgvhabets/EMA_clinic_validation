"""

"""

import numpy as np
from scipy.stats import variation, pearsonr
from scipy.signal import find_peaks, welch
from sklearn.decomposition import PCA


def get_jerk_magn(self,):
    """
    calculate jerk magnitude (derivative of svm) as 
    best approximation for angular velocity
    """

    jerk_mag = np.abs(np.gradient(self.acc_svm))
    jerk_mag = np.mean(jerk_mag)

    return jerk_mag

def get_svm_var(self):

    coef_var = variation(self.acc_svm)

    return coef_var

def get_svm_sd(self):

    stddev = np.std(self.acc_svm)

    return stddev

def get_psd(self):

    fx, psx = welch(
        x=self.acc_svm, fs=self.sfreq,
        nperseg=self.sfreq, noverlap=0,
    )

    return fx, psx

def get_pow_1_4(self):
    """
    selects 1, 2, and 3 Hz
    """

    f_sel = (self.fx >= 1) & (self.fx < 4)
    pow = sum(self.psx[f_sel])

    return pow

def get_pow_4_8(self):

    f_sel = (self.fx >= 4) & (self.fx <= 8)
    pow = sum(self.psx[f_sel])

    return pow

def get_pow_2ndFreq(self, PEAK_WIDTH=1):
    
    f_peaks, peak_props = find_peaks(x=self.psx, height=np.min(self.psx),)
    peak_idx_ordered = np.argsort(peak_props['peak_heights'])  # sorts from small to big
    try:
        f_2nd_peak = f_peaks[peak_idx_ordered[-2]]
    except IndexError:
        print(f_peaks, peak_props, peak_idx_ordered)
        return 0
    
    # print(f'2nd peak found: {f_2nd_peak} Hz \t(after first {f_peaks[peak_idx_ordered[-1]]} Hz)')
    f_sel = (
        self.fx >= f_2nd_peak - PEAK_WIDTH) & (
        self.fx <= (f_2nd_peak + PEAK_WIDTH)
    )
    pow = sum(self.psx[f_sel])

    return pow

def get_dom_freq(self, PEAK_WIDTH=1):
    
    f_peaks, peak_props = find_peaks(x=self.psx, height=np.min(self.psx),)
    i_max_peak = np.argmax(peak_props['peak_heights'])  # sorts from small to big
    dom_freq = f_peaks[i_max_peak]

    return dom_freq

def get_corrcoef_components(self):

    pca = PCA(n_components=3)
    comps = pca.fit_transform(self.acc_triax)
    corr_coef, _ = pearsonr(abs(comps[:, 0]), abs(comps[:, 1]))

    return corr_coef