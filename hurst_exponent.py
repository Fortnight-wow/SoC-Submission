import numpy as np


def _rsj(x):
    """Rescaled range statistic for a segment."""
    mean_x = np.mean(x)
    deviate = np.cumsum(x - mean_x)
    R = np.max(deviate) - np.min(deviate)
    S = np.std(x, ddof=1)
    return R / S if S > 0 else np.nan


def hurst_rs(ts, min_window=10, max_window=None):
    """Hurst exponent via Rescaled Range (R/S) method."""
    ts = np.array(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    N = len(ts)
    if max_window is None:
        max_window = N // 2
    if max_window < min_window:
        return np.nan

    windows, rs_vals = [], []
    step = max(1, (max_window - min_window) // 20)
    for w in range(min_window, max_window + 1, step):
        seg_rs = [_rsj(ts[s:s + w]) for s in range(0, N - w + 1, w)]
        seg_rs = [r for r in seg_rs if not np.isnan(r)]
        if seg_rs:
            windows.append(w)
            rs_vals.append(np.mean(seg_rs))

    if len(windows) < 2:
        return np.nan
    return np.polyfit(np.log(windows), np.log(rs_vals), 1)[0]


def hurst_dfa(ts, min_window=10, max_window=None):
    """Hurst exponent via Detrended Fluctuation Analysis (DFA)."""
    ts = np.array(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    N = len(ts)
    if max_window is None:
        max_window = N // 4
    if max_window < min_window:
        return np.nan

    y = np.cumsum(ts - np.mean(ts))
    windows, fluctuations = [], []
    step = max(1, (max_window - min_window) // 20)
    for w in range(min_window, max_window + 1, step):
        segs = N // w
        if segs < 1:
            continue
        rms = []
        for i in range(segs):
            seg = y[i * w:(i + 1) * w]
            trend = np.polyval(np.polyfit(np.arange(w), seg, 1), np.arange(w))
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms:
            windows.append(w)
            fluctuations.append(np.mean(rms))

    if len(windows) < 2:
        return np.nan
    return np.polyfit(np.log(windows), np.log(fluctuations), 1)[0]


def classify_regime(H):
    """Classify market regime from Hurst exponent."""
    if H < 0.45:
        return "mean_revert", "Use pairs trading"
    elif H > 0.55:
        return "trend", "Use trend-following"
    return "random", "Market appears random"


def calculate_hurst_exponent(series, min_window=10, max_window=None):
    """Calculate Hurst exponent and classify regime."""
    ts = np.array(series, dtype=float)
    ts = ts[~np.isnan(ts)]

    H_rs = hurst_rs(ts, min_window, max_window)
    H_dfa = hurst_dfa(ts, min_window, max_window)

    if not np.isnan(H_rs) and not np.isnan(H_dfa):
        H_combined = (H_rs + H_dfa) / 2.0
    else:
        H_combined = H_rs if not np.isnan(H_rs) else H_dfa

    if np.isnan(H_combined):
        return {'hurst_rs': H_rs, 'hurst_dfa': H_dfa, 'hurst_combined': H_combined,
                'regime': 'unknown', 'recommendation': 'Insufficient data'}

    regime, recommendation = classify_regime(H_combined)
    return {'hurst_rs': H_rs, 'hurst_dfa': H_dfa, 'hurst_combined': H_combined,
            'regime': regime, 'recommendation': recommendation}
