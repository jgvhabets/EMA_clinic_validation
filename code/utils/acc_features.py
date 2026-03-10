"""

"""

import numpy as np
from scipy.stats import variation, kurtosis as scipy_kurtosis, skew as scipy_skew
from scipy.signal import find_peaks, welch, butter, filtfilt
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


def get_pow(self, f_lo, f_hi, include_low=True, include_high=True,
            RATIO_TO_ALL_HZ=False,):
    """
    examples: 

    # get_pow_1_4:
    # (self.fx >= 1) & (self.fx < 4)
    # get_pow(self, 1.0, 4.0, include_low=True,  include_high=True)

    # get_pow_4_8:
    # (self.fx >= 4) & (self.fx <= 8)
    # get_pow(self, 4.0, 8.0, include_low=True,  include_high=True)
    """

    if include_low:
        cond_lo = (self.fx >= f_lo)
    else:
        cond_lo = (self.fx > f_lo)

    if include_high:
        cond_hi = (self.fx <= f_hi)
    else:
        cond_hi = (self.fx < f_hi)

    f_sel = cond_lo & cond_hi

    pow = sum(self.psx[f_sel])

    if RATIO_TO_ALL_HZ:
        pow = pow / sum(self.psx)

    return pow


def get_pow_2ndFreq(self, PEAK_WIDTH=1):

    f_peaks, peak_props = find_peaks(x=self.psx, height=np.min(self.psx),)
    peak_idx_ordered = np.argsort(peak_props['peak_heights'])  # sorts from small to big
    try:
        f_2nd_peak = f_peaks[peak_idx_ordered[-2]]
    except IndexError:
        print(f_peaks, peak_props, peak_idx_ordered)
        return 0

    # band: [f_2nd_peak - PEAK_WIDTH, f_2nd_peak + PEAK_WIDTH]
    f_lo = f_2nd_peak - PEAK_WIDTH
    f_hi = f_2nd_peak + PEAK_WIDTH

    # full closed interval around second peak
    pow = get_pow(self, f_lo, f_hi, include_low=True, include_high=True)

    return pow


def get_dom_freq(self, f_lo=None, f_hi=None,
                 include_low=True, include_high=True):
    """
    examples:

    # 1–4 Hz: (fx >= 1) & (fx < 4):
    # get_dom_freq(self, 1.0, 4.0,
    #              include_low=True, include_high=True)

    # 4–8 Hz: (fx >= 4) & (fx <= 8):
    # get_dom_freq(self, 4.0, 8.0,
    #              include_low=True, include_high=True)
    """

    fx = self.fx
    psx = self.psx

    # full spectrum
    if f_lo is None and f_hi is None:
        sub_fx = fx
        sub_psx = psx
    else:
        # restricted band
        if include_low:
            cond_lo = (fx >= f_lo)
        else:
            cond_lo = (fx > f_lo)

        if include_high:
            cond_hi = (fx <= f_hi)
        else:
            cond_hi = (fx < f_hi)

        mask = cond_lo & cond_hi
        if not np.any(mask):
            return np.nan

        sub_fx = fx[mask]
        sub_psx = psx[mask]

    i_max_peak = np.argmax(sub_psx)
    dom_freq = sub_fx[i_max_peak]

    return dom_freq


def get_corrcoef_components(self):

    pca = PCA(n_components=3)
    comps = pca.fit_transform(self.acc_triax)
    corr_coef = float(np.corrcoef(np.abs(comps[:, 0]), np.abs(comps[:, 1]))[0, 1])

    return corr_coef


def get_mean_acc(self):

    mean_acc = np.mean(self.acc_svm)

    return mean_acc


def get_rms_acc(self):

    x = np.asarray(self.acc_svm, dtype=float)
    rms = np.sqrt(np.mean(x ** 2))

    return rms


def get_range_acc(self):

    x = np.asarray(self.acc_svm, dtype=float)
    range_acc = np.max(x) - np.min(x)

    return range_acc


def get_iqr_acc(self):

    x = np.asarray(self.acc_svm, dtype=float)
    q75 = np.percentile(x, 75.0)
    q25 = np.percentile(x, 25.0)
    iqr = q75 - q25

    return iqr


def get_tremor_power_ratio_3_7_over_3_7_plus_7_12(self):
    """
    3–7 Hz: [3, 7)  -> include_low=True, include_high=True
    7–12 Hz: [7,12] -> include_low=True, include_high=True
    """

    pow_3_7 = get_pow(self, 3.0, 7.0,
                      include_low=True, include_high=True)
    pow_7_12 = get_pow(self, 7.0, 12.0,
                       include_low=True, include_high=True)

    if np.isnan(pow_3_7) or np.isnan(pow_7_12):
        return np.nan

    denom = pow_3_7 + pow_7_12
    if denom <= 0:
        return np.nan

    ratio = pow_3_7 / denom

    return ratio


def get_pow_ratio_to_total(self, f_lo, f_hi,
                           include_low=True, include_high=True):
    """
    examples
    # 4–8 Hz (>=4 & <=8) over total:
    # get_pow_ratio_to_total(self, 4.0, 8.0,
    #                        include_low=True, include_high=True)

    # 3–7 Hz (>=3 & <7) over total:
    # get_pow_ratio_to_total(self, 3.0, 7.0,
    #                        include_low=True, include_high=True)
    """

    band_pow = get_pow(
        self,
        f_lo,
        f_hi,
        include_low=include_low,
        include_high=include_high,
    )
    total_pow = sum(self.psx)

    if np.isnan(band_pow):
        return np.nan

    if total_pow <= 0:
        return np.nan

    ratio = band_pow / total_pow

    return ratio


def get_mean_pow(self, f_lo, f_hi,
                 include_low=True, include_high=True):
    """
    examples:

    # mean power in 0.2–4 Hz:
    # get_mean_pow(self, 0.2, 4.0,
    #              include_low=True, include_high=True)

    # mean power in 3.8 Hz:
    # get_mean_pow(self, 3.0, 8.0,
    #              include_low=True, include_high=True)
    """

    fx = self.fx
    psx = self.psx

    if include_low:
        cond_lo = (fx >= f_lo)
    else:
        cond_lo = (fx > f_lo)

    if include_high:
        cond_hi = (fx <= f_hi)
    else:
        cond_hi = (fx < f_hi)

    mask = cond_lo & cond_hi
    if not np.any(mask):
        return np.nan

    mean_pow = float(np.mean(psx[mask]))

    return mean_pow


def get_peak_pow(self, f_lo, f_hi,
                 include_low=True, include_high=True):
    """
    examples:
    
    # peak power in 0.2–4 Hz:
    # get_peak_pow(self, 0.2, 4.0,
    #              include_low=True, include_high=True)

    # peak power in 3–8 Hz:
    # get_peak_pow(self, 3.0, 8.0,
    #              include_low=True, include_high=True)
    """

    fx = self.fx
    psx = self.psx

    if include_low:
        cond_lo = (fx >= f_lo)
    else:
        cond_lo = (fx > f_lo)

    if include_high:
        cond_hi = (fx <= f_hi)
    else:
        cond_hi = (fx < f_hi)

    mask = cond_lo & cond_hi
    if not np.any(mask):
        return np.nan

    peak_pow = float(np.max(psx[mask]))

    return peak_pow


def get_dominant_energy_ratio(self):

    if not hasattr(self, "psx") or self.psx is None:
        return np.nan

    total_pow = float(np.sum(self.psx))
    if total_pow <= 0:
        return np.nan

    max_pow = float(np.max(self.psx))
    ratio = max_pow / total_pow

    return ratio


def get_energy_dom_pm05(self):

    dom_freq = get_dom_freq(self)
    if dom_freq is None or np.isnan(dom_freq):
        return np.nan

    f_lo = dom_freq - 0.5
    f_hi = dom_freq + 0.5

    energy = get_pow(
        self,
        f_lo,
        f_hi,
        include_low=True,
        include_high=True,
    )

    return energy


def get_log_energy_3_5_7_5(self):

    fx = self.fx
    psx = self.psx

    mask = (fx >= 3.5) & (fx <= 7.5)
    if not np.any(mask):
        return np.nan

    energy = float(np.sum(psx[mask]))
    log_energy = float(np.log(energy + 1e-12))  # avoid log(0)

    return log_energy



def _get_lowpass_acc(self, fc=3.5, order=4):

    if getattr(self, "sfreq", None) is None or self.acc_svm is None:
        return None

    nyq = 0.5 * self.sfreq
    if nyq <= 0 or fc >= nyq:
        return None

    norm_fc = fc / nyq
    b, a = butter(order, norm_fc, btype="low")

    try:
        acc_lp = filtfilt(b, a, self.acc_svm)
    except Exception:
        return None

    return acc_lp


def get_lp_rms_acc(self, fc=3.5):

    acc_lp = _get_lowpass_acc(self, fc=fc)
    if acc_lp is None:
        return np.nan

    rms_lp = float(np.sqrt(np.mean(acc_lp ** 2)))

    return rms_lp


def get_lp_range_acc(self, fc=3.5):

    acc_lp = _get_lowpass_acc(self, fc=fc)
    if acc_lp is None:
        return np.nan

    range_lp = float(np.max(acc_lp) - np.min(acc_lp))

    return range_lp


def _get_lp_psd(self, fc=3.5):

    acc_lp = _get_lowpass_acc(self, fc=fc)
    if acc_lp is None:
        return None

    fx_lp, psx_lp = welch(
        x=acc_lp, fs=self.sfreq,
        nperseg=self.sfreq, noverlap=0,
    )

    return fx_lp, psx_lp


def get_lp_dom_freq(self, fc=3.5):

    res = _get_lp_psd(self, fc=fc)
    if res is None:
        return np.nan

    fx_lp, psx_lp = res
    if not np.any(psx_lp > 0):
        return np.nan

    idx_max = int(np.argmax(psx_lp))
    dom_freq_lp = float(fx_lp[idx_max])

    return dom_freq_lp


def get_lp_dominant_energy_ratio(self, fc=3.5):

    res = _get_lp_psd(self, fc=fc)
    if res is None:
        return np.nan

    fx_lp, psx_lp = res  
    total_lp = float(np.sum(psx_lp))
    if total_lp <= 0:
        return np.nan

    max_lp = float(np.max(psx_lp))
    ratio_lp = max_lp / total_lp

    return ratio_lp



def _norm_xcorr_and_lag(a, b):

    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
    if denom <= 0:
        return np.nan, np.nan

    corr = np.correlate(a0, b0, mode="full") / denom
    idx_max = int(np.argmax(corr))
    lag = idx_max - (len(a) - 1)

    return float(np.max(corr)), float(lag)


def get_max_norm_xcorr(self):

    if not hasattr(self, "acc_triax") or self.acc_triax is None:
        return np.nan

    x = self.acc_triax[:, 0]
    y = self.acc_triax[:, 1]
    z = self.acc_triax[:, 2]

    corr_xy, _ = _norm_xcorr_and_lag(x, y)
    corr_xz, _ = _norm_xcorr_and_lag(x, z)
    corr_yz, _ = _norm_xcorr_and_lag(y, z)

    corrs = np.array([corr_xy, corr_xz, corr_yz], dtype=float)

    if np.all(np.isnan(corrs)):
        return np.nan

    max_corr = float(np.nanmax(corrs))

    return max_corr


def get_lag_at_max_norm_xcorr(self):
 
    if not hasattr(self, "acc_triax") or self.acc_triax is None:
        return np.nan

    x = self.acc_triax[:, 0]
    y = self.acc_triax[:, 1]
    z = self.acc_triax[:, 2]

    corr_xy, lag_xy = _norm_xcorr_and_lag(x, y)
    corr_xz, lag_xz = _norm_xcorr_and_lag(x, z)
    corr_yz, lag_yz = _norm_xcorr_and_lag(y, z)

    corrs = np.array([corr_xy, corr_xz, corr_yz], dtype=float)
    lags  = np.array([lag_xy, lag_xz, lag_yz], dtype=float)

    if np.all(np.isnan(corrs)):
        return np.nan

    idx = int(np.nanargmax(corrs))
    lag_at_max = float(lags[idx])

    return lag_at_max
    
def get_tremor_power_ratio_1_3_over_1_3_plus_3_12(self):
    """
    ratio = pow(1–3) / (pow(1–3) + pow(3–12))
    """

    pow_1_3 = get_pow(self, 1.0, 3.0,
                      include_low=True, include_high=True)
    pow_3_12 = get_pow(self, 3.0, 12.0,
                       include_low=True, include_high=True)

    if np.isnan(pow_1_3) or np.isnan(pow_3_12):
        ratio = np.nan
        return ratio

    denom = pow_1_3 + pow_3_12
    if denom <= 0:
        ratio = np.nan
        return ratio

    ratio = pow_1_3 / denom

    return ratio


def get_tremor_power_ratio_3_12_over_1_3_plus_3_12(self):
    """
    ratio = pow(3–12) / (pow(1–3) + pow(3–12))
    """

    pow_1_3 = get_pow(self, 1.0, 3.0,
                      include_low=True, include_high=True)
    pow_3_12 = get_pow(self, 3.0, 12.0,
                       include_low=True, include_high=True)

    if np.isnan(pow_1_3) or np.isnan(pow_3_12):
        ratio = np.nan
        return ratio

    denom = pow_1_3 + pow_3_12
    if denom <= 0:
        ratio = np.nan
        return ratio

    ratio = pow_3_12 / denom

    return ratio


def get_dom_freq_above_3(self, include_low=False):
    """
    dominant frequency for f > 3 Hz (default) or f >= 3 Hz (include_low=True)

    NOTE: your get_dom_freq requires both f_lo and f_hi,
          so f_hi is set to max(self.fx).
    """

    if not hasattr(self, "fx") or self.fx is None or len(self.fx) == 0:
        dom_freq_above_3 = np.nan
        return dom_freq_above_3

    f_hi = float(np.nanmax(self.fx))
    if not np.isfinite(f_hi):
        dom_freq_above_3 = np.nan
        return dom_freq_above_3

    dom_freq_above_3 = get_dom_freq(
        self,
        3.0,
        f_hi,
        include_low=include_low,
        include_high=True,
    )

    return dom_freq_above_3


def get_peak_pow_dom_freq_above_3(self, include_low=False):
    """
    peak power (max psx) within f > 3 Hz (or >= 3 Hz).
    this corresponds to the power at the dominant peak above 3 Hz.
    """

    if not hasattr(self, "fx") or not hasattr(self, "psx"):
        peak_pow_above_3 = np.nan
        return peak_pow_above_3

    if self.fx is None or self.psx is None:
        peak_pow_above_3 = np.nan
        return peak_pow_above_3

    fx = np.asarray(self.fx)
    psx = np.asarray(self.psx)

    if include_low:
        mask = (fx >= 3.0)
    else:
        mask = (fx > 3.0)

    mask = mask & np.isfinite(fx) & np.isfinite(psx)
    if not np.any(mask):
        peak_pow_above_3 = np.nan
        return peak_pow_above_3

    peak_pow_above_3 = float(np.max(psx[mask]))

    return peak_pow_above_3


def get_peak_pow_dom_freq_above_3_ratio_total(self, include_low=False):
    """
    ratio = peak_pow_above_3 / total_pow
    """

    peak_pow_above_3 = get_peak_pow_dom_freq_above_3(self, include_low=include_low)
    if peak_pow_above_3 is None or np.isnan(peak_pow_above_3):
        ratio = np.nan
        return ratio

    if not hasattr(self, "psx") or self.psx is None:
        ratio = np.nan
        return ratio

    total_pow = float(np.nansum(self.psx))
    if not np.isfinite(total_pow) or total_pow <= 0:
        ratio = np.nan
        return ratio

    ratio = peak_pow_above_3 / total_pow

    return ratio


# ---------------------------------------------------------------------------
# Feature 2 – Mean Absolute Value per axis
# ---------------------------------------------------------------------------

def get_mav_per_axis(self):
    """Mean absolute value of each accelerometer axis (x, y, z)."""
    triax = np.asarray(self.acc_triax, dtype=float)
    mav = np.mean(np.abs(triax), axis=0)
    return float(mav[0]), float(mav[1]), float(mav[2])


# ---------------------------------------------------------------------------
# Feature 3 – Peak Acceleration (SVM)
# ---------------------------------------------------------------------------

def get_peak_acc(self):
    """Maximum instantaneous SVM value in the window."""
    x = np.asarray(self.acc_svm, dtype=float)
    return float(np.max(x))


# ---------------------------------------------------------------------------
# Feature 4 – Jerk RMS (SVM derivative)
# ---------------------------------------------------------------------------

def get_jerk_rms(self):
    """RMS of the first-order time derivative of SVM."""
    x = np.asarray(self.acc_svm, dtype=float)
    jerk = np.gradient(x)
    return float(np.sqrt(np.mean(jerk ** 2)))


# ---------------------------------------------------------------------------
# Feature 5 – Zero-Crossing Rate per axis
# ---------------------------------------------------------------------------

def get_zero_crossing_rate(self):
    """Zero-crossings per second for each axis (x, y, z)."""
    triax = np.asarray(self.acc_triax, dtype=float)
    n = triax.shape[0]
    duration_s = n / self.sfreq
    zcr = []
    for i in range(3):
        col = triax[:, i]
        crossings = int(np.sum(np.diff(np.sign(col)) != 0))
        zcr.append(crossings / duration_s)
    return tuple(zcr)  # (zcr_x, zcr_y, zcr_z)


# ---------------------------------------------------------------------------
# Feature 6 – Approximate Entropy (ApEn)
# ---------------------------------------------------------------------------

def _approx_entropy(x, m=2, r=None, max_n=5000):
    """Compute Approximate Entropy of time series x.

    Uses a KD-tree (Chebyshev metric) to avoid the O(N^2 * m) memory
    allocation of the naive matrix approach. A subsample stride is applied
    when N > max_n as a safety cap for very long windows.
    """
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    N = len(x)

    # Subsample if signal is too long (preserves temporal structure via stride)
    if N > max_n:
        step = N // max_n
        x = x[::step]
        N = len(x)

    if r is None:
        r = 0.2 * np.std(x)
    if N < m + 2 or r <= 0:
        return np.nan

    def _phi(m_val):
        # shape (N - m_val + 1, m_val) — O(N * m) memory
        templates = np.lib.stride_tricks.sliding_window_view(x, m_val)
        tree = cKDTree(templates)
        # Chebyshev norm (p=inf) matches the ApEn definition
        counts = tree.query_ball_point(templates, r=r, p=np.inf,
                                       return_length=True)
        return float(np.mean(np.log(np.asarray(counts, dtype=float)
                                    / len(templates))))

    return float(_phi(m) - _phi(m + 1))


def get_approx_entropy(self, m=2):
    """Approximate Entropy (ApEn) of the SVM signal."""
    x = np.asarray(self.acc_svm, dtype=float)
    return _approx_entropy(x, m=m)


# ---------------------------------------------------------------------------
# Feature 7 – Autocorrelation Lag-1 (SVM)
# ---------------------------------------------------------------------------

def get_autocorr_lag1(self):
    """Normalized lag-1 autocorrelation of the SVM signal."""
    x = np.asarray(self.acc_svm, dtype=float)
    x0 = x - np.mean(x)
    denom = float(np.sum(x0 ** 2))
    if denom <= 0:
        return np.nan
    return float(np.sum(x0[:-1] * x0[1:]) / denom)


# ---------------------------------------------------------------------------
# Feature 10 – Spectral Entropy (SVM)
# ---------------------------------------------------------------------------

def get_spectral_entropy(self):
    """Shannon entropy of the normalized power spectrum."""
    psx = np.asarray(self.psx, dtype=float)
    total = float(np.sum(psx))
    if total <= 0:
        return np.nan
    p = psx / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Feature 11 – Spectral Edge Frequency (95th percentile, SVM)
# ---------------------------------------------------------------------------

def get_spectral_edge_freq(self, percentile=95):
    """Frequency below which `percentile`% of spectral power lies."""
    fx = np.asarray(self.fx, dtype=float)
    psx = np.asarray(self.psx, dtype=float)
    cumulative = np.cumsum(psx)
    total = cumulative[-1]
    if total <= 0:
        return np.nan
    threshold = (percentile / 100.0) * total
    idx = int(np.searchsorted(cumulative, threshold))
    idx = min(idx, len(fx) - 1)
    return float(fx[idx])


# ---------------------------------------------------------------------------
# Feature 12 – Kurtosis (SVM)
# ---------------------------------------------------------------------------

def get_kurtosis(self):
    """Excess kurtosis of the SVM amplitude distribution."""
    x = np.asarray(self.acc_svm, dtype=float)
    return float(scipy_kurtosis(x, fisher=True))


# ---------------------------------------------------------------------------
# Feature 13 – Skewness (SVM)
# ---------------------------------------------------------------------------

def get_skewness(self):
    """Skewness of the SVM amplitude distribution."""
    x = np.asarray(self.acc_svm, dtype=float)
    return float(scipy_skew(x))


# ---------------------------------------------------------------------------
# Feature 14 – Pairwise axis Pearson correlations (x–y, x–z, y–z)
# ---------------------------------------------------------------------------

def get_axis_correlations(self):
    """Pearson correlation between each pair of accelerometer axes.

    Returns (corr_xy, corr_xz, corr_yz).
    """
    if not hasattr(self, 'acc_triax') or self.acc_triax is None:
        return np.nan, np.nan, np.nan
    # np.corrcoef avoids scipy pearsonr incompatibility with numpy >= 2.0
    C = np.corrcoef(np.asarray(self.acc_triax, dtype=float).T)  # shape (3, 3)
    return float(C[0, 1]), float(C[0, 2]), float(C[1, 2])


# ---------------------------------------------------------------------------
# Feature 15 – Ratio of SVM Variance to Gravity Component Variance
# ---------------------------------------------------------------------------

def get_svm_var_to_gravity_var_ratio(self):
    """Ratio of dynamic (SVM) variance to gravity-projection variance.

    Gravity direction is estimated as the unit vector of the mean
    triaxial acceleration. The signal is then projected onto that
    direction; high values indicate active movement relative to posture.
    """
    if not hasattr(self, 'acc_triax') or self.acc_triax is None:
        return np.nan
    triax = np.asarray(self.acc_triax, dtype=float)
    svm_var = float(np.var(self.acc_svm))

    g_vec = np.mean(triax, axis=0)
    g_norm = float(np.linalg.norm(g_vec))
    if g_norm <= 0:
        return np.nan
    g_unit = g_vec / g_norm

    gravity_projection = triax @ g_unit
    gravity_var = float(np.var(gravity_projection))
    if gravity_var <= 0:
        return np.nan

    return svm_var / gravity_var
