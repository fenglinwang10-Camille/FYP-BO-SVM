"""
features.py

PSD, statistical, entropy, Hjorth, and optional wavelet feature extraction.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


EEG_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

_WAVELET_WARNING_PRINTED = False


def _band_power(freqs, psd, low, high):
    """Integrate PSD values inside one frequency band."""
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _shannon_entropy(values, bins=32):
    """Compute Shannon entropy from a histogram of signal amplitudes."""
    hist, _ = np.histogram(values, bins=bins, density=True)
    probabilities = hist / (np.sum(hist) + 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _spectral_entropy(psd):
    """Compute entropy of the normalized power spectrum."""
    probabilities = psd / (np.sum(psd) + 1e-12)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _safe_ratio(numerator, denominator):
    """Return a numerically stable ratio feature."""
    return float(numerator / (denominator + 1e-12))


def _approximate_entropy(values, m=2, r_ratio=0.2, max_points=80):
    """
    Compute a lightweight approximate entropy feature.

    The signal is uniformly downsampled when needed so repeated BO evaluations
    stay practical. This feature captures signal regularity/complexity beyond
    simple PSD and statistical values.
    """
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[indices]

    r = r_ratio * (np.std(values) + 1e-12)

    def _phi(order):
        patterns = np.array(
            [values[index : index + order] for index in range(len(values) - order + 1)]
        )
        if len(patterns) == 0:
            return 0.0

        counts = []
        for pattern in patterns:
            distances = np.max(np.abs(patterns - pattern), axis=1)
            counts.append(np.mean(distances <= r))

        counts = np.asarray(counts, dtype=float)
        return float(np.mean(np.log(counts + 1e-12)))

    return float(_phi(m) - _phi(m + 1))


def _hjorth_parameters(window):
    """
    Compute Hjorth activity, mobility, and complexity.

    These features describe signal variance and shape complexity. They may help
    separate difficult classes such as N and F beyond simple PSD/statistics.
    """
    first_derivative = np.diff(window)
    second_derivative = np.diff(first_derivative)

    activity = float(np.var(window))
    mobility = float(
        np.sqrt(np.var(first_derivative) / (np.var(window) + 1e-12))
    )
    first_mobility = np.sqrt(
        np.var(second_derivative) / (np.var(first_derivative) + 1e-12)
    )
    complexity = float(first_mobility / (mobility + 1e-12))

    return [activity, mobility, complexity]


def _wavelet_features(window, wavelet="db4", level=3, enhanced=False):
    """
    Return wavelet coefficient features.

    PyWavelets is optional. If it is not installed or decomposition fails, the
    pipeline continues with non-wavelet features.
    """
    global _WAVELET_WARNING_PRINTED

    try:
        import pywt

        max_level = pywt.dwt_max_level(len(window), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)
        if level <= 0:
            return []

        coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)
        energies = [float(np.sum(np.asarray(coeff, dtype=float) ** 2)) for coeff in coeffs]
        total_energy = float(np.sum(energies)) + 1e-12
        features = []
        for coeff, energy in zip(coeffs, energies):
            coeff = np.asarray(coeff, dtype=float)
            if enhanced:
                coeff_power = coeff ** 2
                probabilities = coeff_power / (np.sum(coeff_power) + 1e-12)
                probabilities = probabilities[probabilities > 0]
                coeff_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
                features.extend(
                    [
                        energy,
                        float(energy / total_energy),
                        coeff_entropy,
                        float(np.mean(coeff)),
                        float(np.std(coeff)),
                    ]
                )
            else:
                features.extend(
                    [
                        energy,
                        float(np.mean(coeff)),
                        float(np.std(coeff)),
                    ]
                )
        return features
    except Exception as exc:
        if not _WAVELET_WARNING_PRINTED:
            print(
                "Warning: wavelet features were skipped. "
                "Install PyWavelets with: pip install PyWavelets. "
                f"Reason: {exc}"
            )
            _WAVELET_WARNING_PRINTED = True
        return []


def extract_psd_statistical_features(
    windows,
    fs,
    use_entropy=True,
    use_hjorth=True,
    use_wavelet=True,
    use_band_ratios=False,
    enhanced_wavelet=False,
):
    """
    Convert EEG windows into numeric features.

    Existing PSD/statistical features are always kept. Entropy, Hjorth, and
    wavelet features can be switched on/off for feature ablation experiments.
    Richer features may help reduce N/F confusion, but the ablation study checks
    whether they truly help instead of adding everything blindly.
    """
    feature_rows = []

    for window in windows:
        window = np.asarray(window, dtype=float)
        freqs, psd = welch(window, fs=fs, nperseg=min(256, len(window)))

        band_powers = [
            _band_power(freqs, psd, low, high)
            for _, low, high in EEG_BANDS
        ]
        total_power = float(np.sum(band_powers)) + 1e-12
        relative_powers = [power / total_power for power in band_powers]
        band_power_map = {
            band_name: power
            for (band_name, _, _), power in zip(EEG_BANDS, band_powers)
        }
        band_ratio_features = []
        if use_band_ratios:
            band_ratio_features = [
                _safe_ratio(band_power_map["delta"], band_power_map["theta"]),
                _safe_ratio(band_power_map["theta"], band_power_map["alpha"]),
                _safe_ratio(band_power_map["alpha"], band_power_map["beta"]),
                _safe_ratio(band_power_map["beta"], band_power_map["gamma"]),
                _safe_ratio(
                    band_power_map["delta"] + band_power_map["theta"],
                    band_power_map["alpha"] + band_power_map["beta"],
                ),
                _safe_ratio(band_power_map["gamma"], total_power),
            ]

        centered = window - np.mean(window)
        zero_crossings = np.sum(np.diff(np.signbit(centered)) != 0)
        line_length = np.sum(np.abs(np.diff(window)))
        energy = np.sum(window ** 2)

        statistical_features = [
            float(np.mean(window)),
            float(np.std(window)),
            float(np.var(window)),
            float(np.max(window)),
            float(np.min(window)),
            float(np.median(window)),
            float(np.ptp(window)),
            float(np.sqrt(np.mean(window ** 2))),
            float(line_length),
            float(skew(window)),
            float(kurtosis(window)),
            float(energy),
            float(zero_crossings),
        ]

        entropy_features = []
        if use_entropy:
            entropy_features = [
                _shannon_entropy(window),
                _spectral_entropy(psd),
                _approximate_entropy(window),
            ]

        hjorth_features = _hjorth_parameters(window) if use_hjorth else []
        wavelet_features = (
            _wavelet_features(window, enhanced=enhanced_wavelet)
            if use_wavelet
            else []
        )

        feature_rows.append(
            band_powers
            + relative_powers
            + band_ratio_features
            + statistical_features
            + entropy_features
            + hjorth_features
            + wavelet_features
        )

    features = np.asarray(feature_rows, dtype=float)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_by_setting(windows, fs, feature_setting="entropy_hjorth"):
    """
    Extract features using a named feature setting.

    Supported settings:
        original: PSD/statistical features only
        entropy_hjorth: PSD/statistical + entropy + Hjorth
        expanded: PSD/statistical + entropy + Hjorth + wavelet
        nf_enhanced: expanded plus band ratios and wavelet entropy/relative energy
    """
    aliases = {
        "original_features": "original",
        "entropy_hjorth_features": "entropy_hjorth",
        "expanded_features": "expanded",
        "nf_enhanced_features": "nf_enhanced",
    }
    feature_setting = aliases.get(feature_setting, feature_setting)

    if feature_setting == "original":
        return extract_psd_statistical_features(
            windows,
            fs,
            use_entropy=False,
            use_hjorth=False,
            use_wavelet=False,
        )
    if feature_setting == "entropy_hjorth":
        return extract_psd_statistical_features(
            windows,
            fs,
            use_entropy=True,
            use_hjorth=True,
            use_wavelet=False,
        )
    if feature_setting == "expanded":
        return extract_psd_statistical_features(
            windows,
            fs,
            use_entropy=True,
            use_hjorth=True,
            use_wavelet=True,
        )
    if feature_setting == "nf_enhanced":
        return extract_psd_statistical_features(
            windows,
            fs,
            use_entropy=True,
            use_hjorth=True,
            use_wavelet=True,
            use_band_ratios=True,
            enhanced_wavelet=True,
        )

    raise ValueError(f"Unknown feature_setting: {feature_setting}")
