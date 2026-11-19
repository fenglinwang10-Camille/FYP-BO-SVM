"""
preprocessing.py

Signal preprocessing helpers for the Bonn EEG SVM pipeline.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter_signal(signal, fs, f_low, f_high, order=4):
    """Apply a Butterworth band-pass filter to one EEG signal."""
    nyquist = fs / 2.0
    low = f_low / nyquist
    high = f_high / nyquist

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


def bandpass_filter_signals(raw_signals, fs, f_low, f_high):
    """Apply the same band-pass filter to all EEG recordings."""
    return [
        bandpass_filter_signal(signal, fs, f_low, f_high)
        for signal in raw_signals
    ]


def segment_windows(signals, labels, fs, window_size, overlap=0.0):
    """
    Split each EEG recording into fixed-length windows.

    The class label of the original recording is copied to each window.
    overlap is the fractional overlap between adjacent windows. For example,
    overlap=0.5 means the next window starts halfway through the current one.
    """
    samples_per_window = int(round(window_size * fs))
    if samples_per_window <= 0:
        raise ValueError("window_size is too small for the sampling rate.")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in the range [0, 1).")

    step = max(1, int(round(samples_per_window * (1.0 - overlap))))

    windows = []
    window_labels = []

    for signal, label in zip(signals, labels):
        signal = np.asarray(signal, dtype=float)

        for start in range(0, len(signal) - samples_per_window + 1, step):
            end = start + samples_per_window
            windows.append(signal[start:end])
            window_labels.append(label)

    if not windows:
        raise ValueError(
            "No windows were created. Try a smaller window_size or check signal length."
        )

    return np.asarray(windows, dtype=float), np.asarray(window_labels, dtype=int)
