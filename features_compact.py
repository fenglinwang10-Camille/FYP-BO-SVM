"""Compact 32-feature EEG window representation (vectorised).

New feature module for a possible protocol v2. It does not replace
features.py, whose hash is frozen in the completed experiments.

Design choices:
* power-like quantities are log-transformed, so RBF distances are not
  dominated by amplitude differences of several orders of magnitude;
* sums are normalised per second (line length, zero crossings), so their
  scale does not change with the searched window length;
* approximate entropy is replaced by permutation entropy, which uses every
  sample and needs no down-sampling;
* each amplitude measure appears once (log variance) instead of six times.

Columns (32):
 0-4   log absolute band power  delta, theta, alpha, beta, gamma(30-40 Hz)
 5-9   relative band power      same bands
 10-14 log DWT energy           db4 level 4: A4, D4, D3, D2, D1
 15-19 log DWT std              same sub-bands
 20    permutation entropy      order 4, delay 1, normalised to [0, 1]
 21    Higuchi fractal dim.     kmax 10
 22    Katz fractal dim.
 23-24 Hjorth mobility, complexity
 25    line length per second
 26    zero crossings per second
 27    spectral edge frequency 90 % (0.5-40 Hz)
 28    spectral entropy (0.5-40 Hz, natural log)
 29-30 skewness, excess kurtosis
 31    log variance
"""

from math import factorial

import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))
WAVELET, WAVELET_LEVEL = "db4", 4
PE_ORDER, HIGUCHI_KMAX = 4, 10
EPS = 1e-12
FEATURE_NAMES = (
    [f"log_bp_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")]
    + [f"rel_bp_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")]
    + [f"log_dwt_energy_{s}" for s in ("A4", "D4", "D3", "D2", "D1")]
    + [f"log_dwt_std_{s}" for s in ("A4", "D4", "D3", "D2", "D1")]
    + ["perm_entropy", "higuchi_fd", "katz_fd", "hjorth_mobility", "hjorth_complexity",
       "line_length_per_s", "zero_crossings_per_s", "sef90", "spectral_entropy",
       "skewness", "kurtosis", "log_variance"]
)
N_FEATURES = len(FEATURE_NAMES)


def _spectral(windows, fs):
    freqs, psd = welch(windows, fs=fs, nperseg=min(256, windows.shape[1]), axis=-1)
    band_power = np.stack([
        np.trapezoid(psd[:, (freqs >= lo) & (freqs < hi)], freqs[(freqs >= lo) & (freqs < hi)], axis=1)
        for lo, hi in BANDS], axis=1)
    relative = band_power / (band_power.sum(axis=1, keepdims=True) + EPS)
    in_band = (freqs >= 0.5) & (freqs <= 40.0)
    f_band, p_band = freqs[in_band], psd[:, in_band]
    cumulative = np.cumsum(p_band, axis=1)
    edge_index = np.argmax(cumulative >= 0.9 * cumulative[:, -1:], axis=1)
    sef90 = f_band[edge_index]
    normalised = p_band / p_band.sum(axis=1, keepdims=True)
    spectral_entropy = -np.sum(normalised * np.log(normalised + 1e-20), axis=1)
    return np.log(band_power + EPS), relative, sef90, spectral_entropy


def _wavelet(windows):
    max_level = pywt.dwt_max_level(windows.shape[1], pywt.Wavelet(WAVELET).dec_len)
    if max_level < WAVELET_LEVEL:
        raise ValueError(f"Window of {windows.shape[1]} samples is too short for level {WAVELET_LEVEL}")
    coefficients = pywt.wavedec(windows, WAVELET, level=WAVELET_LEVEL, axis=-1)
    energy = np.stack([np.log(np.sum(c ** 2, axis=1) + EPS) for c in coefficients], axis=1)
    spread = np.stack([np.log(np.std(c, axis=1) + EPS) for c in coefficients], axis=1)
    return energy, spread


def permutation_entropy(windows, order=PE_ORDER):
    embedded = np.lib.stride_tricks.sliding_window_view(windows, order, axis=1)
    codes = np.argsort(embedded, axis=2) @ (order ** np.arange(order))
    n_codes = order ** order
    offsets = np.arange(len(windows))[:, None] * n_codes
    counts = np.bincount((codes + offsets).ravel(), minlength=len(windows) * n_codes)
    p = counts.reshape(len(windows), n_codes) / codes.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=1) / np.log(factorial(order))


def higuchi_fd(windows, kmax=HIGUCHI_KMAX):
    n = windows.shape[1]
    ks = np.arange(1, kmax + 1)
    curve_length = []
    for k in ks:
        lengths = []
        for m in range(k):
            sub = windows[:, m::k]
            if sub.shape[1] < 2:
                continue
            lengths.append(np.abs(np.diff(sub, axis=1)).sum(axis=1) * (n - 1) / ((sub.shape[1] - 1) * k) / k)
        curve_length.append(np.mean(lengths, axis=0))
    log_l = np.log(np.stack(curve_length, axis=1))
    x = np.log(1.0 / ks)
    x_centred = x - x.mean()
    return (log_l - log_l.mean(axis=1, keepdims=True)) @ x_centred / np.sum(x_centred ** 2)


def katz_fd(windows):
    length = np.abs(np.diff(windows, axis=1)).sum(axis=1)
    diameter = np.max(np.abs(windows - windows[:, :1]), axis=1)
    steps = windows.shape[1] - 1
    return np.log10(steps) / (np.log10(steps) + np.log10(diameter / length + EPS))


def extract_compact_features(windows, fs):
    """Return an (n_windows, 32) feature matrix for equal-length windows."""
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2:
        raise ValueError("windows must be a 2-D array (n_windows, n_samples)")
    seconds = windows.shape[1] / fs
    log_bp, relative, sef90, spectral_entropy = _spectral(windows, fs)
    dwt_energy, dwt_std = _wavelet(windows)
    d1 = np.diff(windows, axis=1)
    d2 = np.diff(d1, axis=1)
    var0, var1, var2 = windows.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    centred = windows - windows.mean(axis=1, keepdims=True)
    zero_crossings = np.sum(np.diff(np.signbit(centred), axis=1), axis=1) / seconds
    features = np.column_stack([
        log_bp, relative, dwt_energy, dwt_std,
        permutation_entropy(windows), higuchi_fd(windows), katz_fd(windows),
        mobility, complexity, np.abs(d1).sum(axis=1) / seconds, zero_crossings,
        sef90, spectral_entropy, skew(windows, axis=1), kurtosis(windows, axis=1),
        np.log(var0 + EPS),
    ])
    assert features.shape[1] == N_FEATURES
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
