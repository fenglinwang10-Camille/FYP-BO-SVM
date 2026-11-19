"""
Independent experiment 2 for the Bonn EEG BO-SVM project.

Pipeline:
raw EEG segment -> fixed 50 Hz notch -> BO-selected 4th-order Butterworth
band-pass -> windowing -> handcrafted feature extraction -> grouped CV /
held-out test evaluation.

This experiment keeps the earlier 35-feature experiment untouched and writes
all new artifacts into its own experiment directory.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, freqz, iirnotch, sosfiltfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loader import load_bonn_dataset
from display_utils import display_dataframe, display_image
from metrics import classification_metrics_summary


EXP_NAME = "experiment_2_notch_extended_handcrafted_features"
EXP_ROOT = SCRIPT_DIR
RESULTS_DIR = EXP_ROOT / "results"
FIGURES_DIR = EXP_ROOT / "figures"
CACHED_DIR = EXP_ROOT / "cached_features"
LOGS_DIR = EXP_ROOT / "logs"

DATASET_PATH = PROJECT_DIR / "Dataset"
FS = 173.61
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_CALLS = 50
N_INITIAL_POINTS = 16
CLASS_WEIGHT = "balanced"
WINDOW_OVERLAP = 0.0
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
BUTTER_ORDER = 4
EPSILON = 1e-12

CLASS_NAMES = ["Z", "O", "N", "F", "S"]
CLASS_LABELS = np.arange(len(CLASS_NAMES), dtype=int)

CANONICAL_BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 70.0),
]

CURRENT_35_FEATURE_NAMES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "relative_gamma_power",
    "delta_theta_ratio",
    "theta_alpha_ratio",
    "alpha_beta_ratio",
    "beta_gamma_ratio",
    "low_high_ratio",
    "gamma_total_ratio",
    "mean",
    "standard_deviation",
    "variance",
    "maximum",
    "minimum",
    "median",
    "peak_to_peak_amplitude",
    "root_mean_square",
    "line_length",
    "skewness",
    "kurtosis",
    "energy",
    "zero_crossing_rate",
    "shannon_entropy",
    "spectral_entropy",
    "approximate_entropy",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]

EXTENDED_FEATURE_NAMES = [
    "interquartile_range",
    "mean_absolute_deviation",
    "dominant_frequency",
    "median_frequency",
    "spectral_edge_frequency_90",
    "spectral_centroid",
    "sample_entropy",
    "permutation_entropy",
]

FINAL_FEATURE_NAMES = CURRENT_35_FEATURE_NAMES + EXTENDED_FEATURE_NAMES

SEARCH_SPACE = [
    Real(0.0, 1.0, name="f_low_u"),
    Real(0.0, 1.0, name="f_high_u"),
    Real(0.0, 1.0, name="window_u"),
    Real(-1.0, 2.0, name="log10_C"),
    Real(-4.0, 0.0, name="log10_gamma"),
]


def _ensure_directories():
    for path in [RESULTS_DIR, FIGURES_DIR, CACHED_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _setup_logging():
    log_path = LOGS_DIR / f"{EXP_NAME}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    return name.upper().replace("_", " ").replace("-", " ").strip()


def _safe_skew(values: np.ndarray) -> float:
    if np.std(values) < EPSILON:
        return 0.0
    value = skew(values, bias=False)
    return float(value) if np.isfinite(value) else 0.0


def _safe_kurtosis(values: np.ndarray) -> float:
    if np.std(values) < EPSILON:
        return 0.0
    value = kurtosis(values, bias=False)
    return float(value) if np.isfinite(value) else 0.0


def _safe_zero_crossing_rate(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    crossings = np.sum(np.diff(np.signbit(centered)) != 0)
    return float(crossings / max(len(values) - 1, 1))


def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _band_power_with_overlap(freqs, psd, band_low, band_high, f_low, f_high):
    lower = max(band_low, f_low)
    upper = min(band_high, f_high)
    if upper <= lower:
        return 0.0, (lower, upper)
    return _band_power(freqs, psd, lower, upper), (lower, upper)


def _sample_entropy(values: np.ndarray, m: int = 2, r_ratio: float = 0.2, max_points: int = 120) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[indices]

    if len(values) <= m + 2:
        return 0.0

    r = r_ratio * (np.std(values) + EPSILON)

    def _phi(order: int) -> float:
        patterns = np.array(
            [values[index : index + order] for index in range(len(values) - order + 1)]
        )
        if len(patterns) <= 1:
            return 0.0
        matches = []
        for pattern in patterns:
            distances = np.max(np.abs(patterns - pattern), axis=1)
            matches.append(np.mean(distances <= r))
        matches = np.asarray(matches, dtype=float)
        return float(np.mean(matches))

    a = _phi(m)
    b = _phi(m + 1)
    if a <= EPSILON or b <= EPSILON:
        return 0.0
    return float(-np.log(b / a))


def _permutation_entropy(values: np.ndarray, order: int = 3, delay: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < order * delay:
        return 0.0

    patterns = []
    for start in range(n - delay * (order - 1)):
        window = values[start : start + delay * order : delay]
        if len(window) == order:
            patterns.append(tuple(np.argsort(window, kind="mergesort")))

    if not patterns:
        return 0.0

    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / np.sum(counts)
    return float(-np.sum(probabilities * np.log2(probabilities + EPSILON)))


def _spectral_entropy(psd: np.ndarray) -> float:
    probabilities = psd / (np.sum(psd) + EPSILON)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    if len(psd) == 0:
        return 0.0
    return float(freqs[int(np.argmax(psd))])


def _median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    if len(psd) == 0:
        return 0.0
    cumulative = np.cumsum(psd)
    target = cumulative[-1] / 2.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(freqs) - 1)
    return float(freqs[index])


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, percentile: float = 0.90) -> float:
    if len(psd) == 0:
        return 0.0
    cumulative = np.cumsum(psd)
    target = cumulative[-1] * percentile
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(freqs) - 1)
    return float(freqs[index])


def _spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    total = float(np.sum(psd)) + EPSILON
    return float(np.sum(freqs * psd) / total)


def _mean_absolute_deviation(values: np.ndarray) -> float:
    mean = np.mean(values)
    return float(np.mean(np.abs(values - mean)))


def _interquartile_range(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def _class_from_path(file_path: Path, dataset_path: Path):
    stem = file_path.stem
    file_prefix = stem[:1].upper()
    if file_prefix in CLASS_NAMES:
        return CLASS_NAMES.index(file_prefix)

    relative_parts = file_path.relative_to(dataset_path).parts[:-1]
    for part in reversed(relative_parts):
        normalised = _normalize_name(part)
        compact = normalised.replace(" ", "")
        if normalised in CLASS_NAMES:
            return CLASS_NAMES.index(normalised)
        if compact in {"SETA", "SETB", "SETC", "SETD", "SETE"}:
            return {"SETA": 0, "SETB": 1, "SETC": 2, "SETD": 3, "SETE": 4}[compact]
    return None


def _load_bonn_dataset(dataset_path: Path):
    signals, labels = load_bonn_dataset(dataset_path, classification="five-class")
    return [np.asarray(signal, dtype=float) for signal in signals], np.asarray(labels, dtype=int)


def _split_original_segments(raw_labels):
    segment_ids = np.arange(len(raw_labels), dtype=int)
    train_ids, test_ids = train_test_split(
        segment_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=raw_labels,
    )
    return np.sort(train_ids), np.sort(test_ids)


def _select_segments(raw_signals, raw_labels, segment_ids):
    signals = [raw_signals[index] for index in segment_ids]
    labels = np.asarray(raw_labels, dtype=int)[segment_ids]
    return signals, labels, np.asarray(segment_ids, dtype=int)


def _design_notch_filter():
    b, a = iirnotch(NOTCH_FREQ, NOTCH_Q, FS)
    return b, a


def _apply_notch_filter(signals):
    b, a = _design_notch_filter()
    return [filtfilt(b, a, np.asarray(signal, dtype=float)) for signal in signals]


def _apply_bandpass_filter(signals, f_low, f_high):
    nyquist = FS / 2.0
    low = f_low / nyquist
    high = f_high / nyquist
    sos = butter(BUTTER_ORDER, [low, high], btype="bandpass", output="sos")
    return [sosfiltfilt(sos, np.asarray(signal, dtype=float)) for signal in signals]


def _segment_with_groups(signals, labels, segment_ids, window_size):
    samples_per_window = int(round(float(window_size) * FS))
    if samples_per_window <= 0:
        raise ValueError("window_size produces no samples")
    if WINDOW_OVERLAP < 0 or WINDOW_OVERLAP >= 1:
        raise ValueError("WINDOW_OVERLAP must be in [0, 1).")
    step = max(1, int(round(samples_per_window * (1.0 - WINDOW_OVERLAP))))

    windows = []
    window_labels = []
    window_groups = []
    for signal, label, segment_id in zip(signals, labels, segment_ids):
        signal = np.asarray(signal, dtype=float)
        for start in range(0, len(signal) - samples_per_window + 1, step):
            windows.append(signal[start : start + samples_per_window])
            window_labels.append(int(label))
            window_groups.append(int(segment_id))

    if not windows:
        raise ValueError("No windows were created.")
    return (
        np.asarray(windows, dtype=float),
        np.asarray(window_labels, dtype=int),
        np.asarray(window_groups, dtype=int),
    )


def _window_psd(window: np.ndarray):
    window = np.asarray(window, dtype=float)
    freqs, psd = welch(window, fs=FS, nperseg=min(256, len(window)))
    return freqs, psd


def _extract_window_features(window: np.ndarray, f_low: float, f_high: float):
    freqs, psd = _window_psd(window)

    band_results = []
    effective_boundaries = {}
    band_power_values = {}
    for band_name, band_low, band_high in CANONICAL_BANDS:
        power, boundaries = _band_power_with_overlap(freqs, psd, band_low, band_high, f_low, f_high)
        band_power_values[band_name] = power
        effective_boundaries[band_name] = boundaries
        band_results.append(power)

    total_band_power = float(np.sum(band_results)) + EPSILON
    relative_powers = [power / total_band_power for power in band_results]

    band_ratio_features = [
        band_power_values["delta"] / (band_power_values["theta"] + EPSILON),
        band_power_values["theta"] / (band_power_values["alpha"] + EPSILON),
        band_power_values["alpha"] / (band_power_values["beta"] + EPSILON),
        band_power_values["beta"] / (band_power_values["gamma"] + EPSILON),
        (band_power_values["delta"] + band_power_values["theta"]) / (band_power_values["alpha"] + band_power_values["beta"] + EPSILON),
        band_power_values["gamma"] / total_band_power,
    ]

    mean_value = float(np.mean(window))
    std_value = float(np.std(window))
    variance_value = float(np.var(window))
    max_value = float(np.max(window))
    min_value = float(np.min(window))
    median_value = float(np.median(window))
    ptp_value = float(np.ptp(window))
    rms_value = float(np.sqrt(np.mean(window ** 2)))
    line_length_value = float(np.sum(np.abs(np.diff(window))))
    energy_value = float(np.sum(window ** 2))
    zero_crossings = _safe_zero_crossing_rate(window)

    spectral_features = [
        _dominant_frequency(freqs, psd),
        _median_frequency(freqs, psd),
        _spectral_edge_frequency(freqs, psd, percentile=0.90),
        _spectral_centroid(freqs, psd),
    ]

    hist, _ = np.histogram(window, bins=32, density=True)
    probabilities = hist / (np.sum(hist) + EPSILON)
    probabilities = probabilities[probabilities > 0]
    shannon_entropy = float(-np.sum(probabilities * np.log2(probabilities + EPSILON)))
    spectral_entropy = _spectral_entropy(psd)
    approximate_entropy = _approximate_entropy(window)
    sample_entropy = _sample_entropy(window)
    permutation_entropy = _permutation_entropy(window)

    hjorth_activity = variance_value
    first_derivative = np.diff(window)
    second_derivative = np.diff(first_derivative)
    mobility = float(np.sqrt(np.var(first_derivative) / (variance_value + EPSILON)))
    first_mobility = float(np.sqrt(np.var(second_derivative) / (np.var(first_derivative) + EPSILON)))
    complexity = float(first_mobility / (mobility + EPSILON))

    interquartile_range = _interquartile_range(window)
    mean_absolute_deviation = _mean_absolute_deviation(window)

    feature_values = [
        *band_results,
        *relative_powers,
        *band_ratio_features,
        mean_value,
        std_value,
        variance_value,
        max_value,
        min_value,
        median_value,
        ptp_value,
        rms_value,
        line_length_value,
        _safe_skew(window),
        _safe_kurtosis(window),
        energy_value,
        zero_crossings,
        shannon_entropy,
        spectral_entropy,
        approximate_entropy,
        hjorth_activity,
        mobility,
        complexity,
        interquartile_range,
        mean_absolute_deviation,
        *spectral_features[:4],
        sample_entropy,
        permutation_entropy,
    ]

    feature_names = FINAL_FEATURE_NAMES
    return feature_names, feature_values, effective_boundaries


def _approximate_entropy(values: np.ndarray, m: int = 2, r_ratio: float = 0.2, max_points: int = 120) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[indices]

    if len(values) <= m + 2:
        return 0.0

    r = r_ratio * (np.std(values) + EPSILON)

    def _phi(order):
        patterns = np.array(
            [values[index : index + order] for index in range(len(values) - order + 1)]
        )
        if len(patterns) <= 1:
            return 0.0
        counts = []
        for pattern in patterns:
            distances = np.max(np.abs(patterns - pattern), axis=1)
            counts.append(np.mean(distances <= r))
        counts = np.asarray(counts, dtype=float)
        return float(np.mean(np.log(counts + EPSILON)))

    return float(_phi(m) - _phi(m + 1))


def _extract_feature_matrix(windows, f_low, f_high):
    feature_rows = []
    effective_boundaries = None
    for window in windows:
        feature_names, feature_values, boundaries = _extract_window_features(window, f_low, f_high)
        feature_rows.append(feature_values)
        if effective_boundaries is None:
            effective_boundaries = boundaries
    matrix = np.asarray(feature_rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature extraction produced NaN or infinity values.")
    return feature_names, matrix, effective_boundaries


def _validate_feature_matrix(matrix, feature_names):
    nan_counts = np.isnan(matrix).sum(axis=0)
    inf_counts = np.isinf(matrix).sum(axis=0)
    return pd.DataFrame(
        {
            "feature_name": feature_names,
            "nan_count": nan_counts.astype(int),
            "inf_count": inf_counts.astype(int),
        }
    )


def _majority_vote_by_segment(predictions, y, groups):
    segment_true = []
    segment_pred = []
    for segment_id in np.unique(groups):
        mask = groups == segment_id
        true_values = np.unique(y[mask])
        if len(true_values) != 1:
            raise RuntimeError("One segment contains more than one true class")
        counts = np.bincount(predictions[mask], minlength=len(CLASS_NAMES))
        segment_true.append(int(true_values[0]))
        segment_pred.append(int(np.argmax(counts)))
    return np.asarray(segment_true), np.asarray(segment_pred)


def _make_model(params):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(params["C"]),
                    gamma=float(params["gamma"]),
                    class_weight=CLASS_WEIGHT,
                ),
            ),
        ]
    )


def _objective_score(metrics):
    return (
        0.35 * metrics["accuracy"]
        + 0.25 * metrics["macro_recall"]
        + 0.20 * metrics["per_class_recall"]["N"]
        + 0.20 * metrics["per_class_recall"]["F"]
    )


def _build_feature_matrix(signals, labels, segment_ids, params):
    notched = _apply_notch_filter(signals)
    filtered = _apply_bandpass_filter(notched, float(params["f_low"]), float(params["f_high"]))
    windows, y, groups = _segment_with_groups(
        filtered,
        labels,
        segment_ids,
        float(params["window_size"]),
    )
    feature_names, X, effective_boundaries = _extract_feature_matrix(
        windows,
        float(params["f_low"]),
        float(params["f_high"]),
    )
    return X, y, groups, windows, feature_names, effective_boundaries


def _cross_validate_training_segments(signals, labels, segment_ids, params):
    X, y, groups, windows, feature_names, effective_boundaries = _build_feature_matrix(
        signals, labels, segment_ids, params
    )
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []
    fold_window_metrics = []
    fold_segment_counts = []

    for train_index, validation_index in splitter.split(X, y, groups):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups.intersection(validation_groups):
            raise RuntimeError("Grouped CV leakage detected")

        model = _make_model(params)
        model.fit(X[train_index], y[train_index])
        predictions = model.predict(X[validation_index])

        window_metrics = classification_metrics_summary(
            y[validation_index],
            predictions,
            CLASS_NAMES,
        )
        segment_true, segment_predictions = _majority_vote_by_segment(
            predictions,
            y[validation_index],
            groups[validation_index],
        )
        segment_metrics = classification_metrics_summary(
            segment_true,
            segment_predictions,
            CLASS_NAMES,
        )
        segment_metrics["macro_f1"] = float(
            f1_score(segment_true, segment_predictions, average="macro", zero_division=0)
        )

        fold_window_metrics.append(window_metrics)
        fold_metrics.append(segment_metrics)
        fold_segment_counts.append(
            {
                "train": len(train_groups),
                "validation": len(validation_groups),
            }
        )

    averaged = {
        "accuracy": float(np.mean([metrics["accuracy"] for metrics in fold_metrics])),
        "macro_recall": float(np.mean([metrics["macro_recall"] for metrics in fold_metrics])),
        "macro_f1": float(np.mean([metrics["macro_f1"] for metrics in fold_metrics])),
        "per_class_recall": {
            name: float(np.mean([metrics["per_class_recall"][name] for metrics in fold_metrics]))
            for name in CLASS_NAMES
        },
    }

    feature_report = _validate_feature_matrix(X, feature_names)
    return _objective_score(averaged), averaged, X.shape, fold_segment_counts, feature_report, feature_names, effective_boundaries, windows, groups, y


def _json_metrics(metrics):
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_recall": {
            name: float(value) for name, value in metrics["per_class_recall"].items()
        },
        "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
    }


def _load_shared_data():
    raw_signals, raw_labels = _load_bonn_dataset(DATASET_PATH)
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids = _split_original_segments(raw_labels)
    train_data = _select_segments(raw_signals, raw_labels, train_ids)
    test_data = _select_segments(raw_signals, raw_labels, test_ids)
    if set(train_ids.tolist()).intersection(set(test_ids.tolist())):
        raise RuntimeError("Outer split overlap detected")
    return raw_signals, raw_labels, train_data, test_data, train_ids, test_ids


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_average_psd(signals, labels, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_signals = [signal for signal, label in zip(signals, labels) if label == class_index]
        freqs_list = []
        psd_list = []
        for signal in class_signals:
            freqs, psd = welch(signal, fs=FS, nperseg=min(1024, len(signal)), noverlap=512)
            freqs_list.append(freqs)
            psd_list.append(psd)
        ax.plot(freqs_list[0], np.mean(np.asarray(psd_list), axis=0), label=class_name)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average PSD")
    ax.set_title(title)
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_psd_zoom(before_signals, before_labels, after_signals, after_labels, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for class_index, class_name in enumerate(CLASS_NAMES):
        before_class = [signal for signal, label in zip(before_signals, before_labels) if label == class_index]
        after_class = [signal for signal, label in zip(after_signals, after_labels) if label == class_index]
        freqs_before, psd_before = welch(before_class[0], fs=FS, nperseg=min(1024, len(before_class[0])), noverlap=512)
        freq_mask = (freqs_before >= 45.0) & (freqs_before <= 55.0)
        before_stack = []
        after_stack = []
        for signal in before_class:
            freqs, psd = welch(signal, fs=FS, nperseg=min(1024, len(signal)), noverlap=512)
            before_stack.append(psd[freq_mask])
        for signal in after_class:
            freqs, psd = welch(signal, fs=FS, nperseg=min(1024, len(signal)), noverlap=512)
            after_stack.append(psd[freq_mask])
        ax.plot(freqs_before[freq_mask], np.mean(before_stack, axis=0), linestyle="--", label=f"{class_name} before")
        ax.plot(freqs_before[freq_mask], np.mean(after_stack, axis=0), linewidth=2, label=f"{class_name} after")

    ax.axvspan(48.0, 52.0, color="#f59e0b", alpha=0.12, label="48-52 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average PSD")
    ax.set_title("Before/After 50 Hz Notch: 45-55 Hz Zoom")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_notch_response(output_path: Path):
    b, a = _design_notch_filter()
    w, h = freqz(b, a, worN=4096, fs=FS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), EPSILON)))
    ax.axvline(NOTCH_FREQ, color="red", linestyle="--", linewidth=1.5, label="50 Hz")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("50 Hz Notch Frequency Response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _summarise_notch_psd(before_signals, before_labels, after_signals, after_labels):
    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        before_class = [signal for signal, label in zip(before_signals, before_labels) if label == class_index]
        after_class = [signal for signal, label in zip(after_signals, after_labels) if label == class_index]

        before_freqs, before_psd = welch(before_class[0], fs=FS, nperseg=min(1024, len(before_class[0])), noverlap=512)
        after_freqs, after_psd = welch(after_class[0], fs=FS, nperseg=min(1024, len(after_class[0])), noverlap=512)

        before_values = []
        after_values = []
        for signal in before_class:
            freqs, psd = welch(signal, fs=FS, nperseg=min(1024, len(signal)), noverlap=512)
            idx = int(np.argmin(np.abs(freqs - NOTCH_FREQ)))
            before_values.append(float(psd[idx]))
        for signal in after_class:
            freqs, psd = welch(signal, fs=FS, nperseg=min(1024, len(signal)), noverlap=512)
            idx = int(np.argmin(np.abs(freqs - NOTCH_FREQ)))
            after_values.append(float(psd[idx]))

        rows.append(
            {
                "class": class_name,
                "psd_50hz_before_mean": float(np.mean(before_values)),
                "psd_50hz_after_mean": float(np.mean(after_values)),
                "suppression_ratio_after_over_before": float(np.mean(after_values) / (np.mean(before_values) + EPSILON)),
                "before_peak_50hz": float(before_psd[int(np.argmin(np.abs(before_freqs - NOTCH_FREQ)))]),
                "after_peak_50hz": float(after_psd[int(np.argmin(np.abs(after_freqs - NOTCH_FREQ)))]),
            }
        )
    return pd.DataFrame(rows)


def _search_space_to_params(decoded_values):
    f_low = 0.5 + decoded_values[0] * (3.0 - 0.5)
    f_high = 30.0 + decoded_values[1] * (70.0 - 30.0)
    window_size = 1.0 + decoded_values[2] * (8.0 - 1.0)
    C = 10 ** decoded_values[3]
    gamma = 10 ** decoded_values[4]
    return {
        "f_low": float(f_low),
        "f_high": float(f_high),
        "window_size": float(window_size),
        "C": float(C),
        "gamma": float(gamma),
    }


def _sample_random_params(rng):
    return {
        "f_low_u": float(rng.uniform(0.0, 1.0)),
        "f_high_u": float(rng.uniform(0.0, 1.0)),
        "window_u": float(rng.uniform(0.0, 1.0)),
        "log10_C": float(rng.uniform(-1.0, 2.0)),
        "log10_gamma": float(rng.uniform(-4.0, 0.0)),
    }


def _flatten_history(iteration, params, score, metrics, feature_shape, feature_report, feature_names, boundaries):
    row = {
        "iteration": iteration,
        "score": float(score),
        **params,
        "class_weight": CLASS_WEIGHT,
        "window_overlap": WINDOW_OVERLAP,
        "cv_folds": CV_FOLDS,
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_macro_recall": float(metrics["macro_recall"]),
        "validation_macro_f1": float(metrics["macro_f1"]),
        "feature_rows": int(feature_shape[0]),
        "feature_columns": int(feature_shape[1]),
        "feature_names": json.dumps(feature_names),
        "band_boundaries": json.dumps(boundaries),
    }
    for class_name in CLASS_NAMES:
        row[f"validation_recall_{class_name}"] = float(metrics["per_class_recall"][class_name])
    row["nan_counts"] = json.dumps(feature_report["nan_count"].to_list())
    row["inf_counts"] = json.dumps(feature_report["inf_count"].to_list())
    return row


def _final_test_once(train_signals, train_labels, train_segment_ids, test_signals, test_labels, test_segment_ids, params):
    X_train, y_train, train_groups, train_windows, feature_names, boundaries = _build_feature_matrix(
        train_signals,
        train_labels,
        train_segment_ids,
        params,
    )
    X_test, y_test, test_groups, test_windows, _, _ = _build_feature_matrix(
        test_signals,
        test_labels,
        test_segment_ids,
        params,
    )

    if set(train_groups.tolist()).intersection(set(test_groups.tolist())):
        raise RuntimeError("Outer train/test segment leakage detected")

    model = _make_model(params)
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    window_metrics = classification_metrics_summary(y_test, test_predictions, CLASS_NAMES)
    segment_true, segment_predictions = _majority_vote_by_segment(test_predictions, y_test, test_groups)
    segment_metrics = classification_metrics_summary(segment_true, segment_predictions, CLASS_NAMES)

    train_validation_report = _validate_feature_matrix(X_train, feature_names)
    test_validation_report = _validate_feature_matrix(X_test, feature_names)
    return (
        window_metrics,
        segment_metrics,
        X_train.shape,
        X_test.shape,
        feature_names,
        boundaries,
        train_windows,
        test_windows,
        train_validation_report,
        test_validation_report,
    )


def _save_feature_audit():
    audit = {
        "original_feature_names": CURRENT_35_FEATURE_NAMES,
        "original_feature_count": len(CURRENT_35_FEATURE_NAMES),
        "new_feature_names": EXTENDED_FEATURE_NAMES,
        "new_feature_count": len(EXTENDED_FEATURE_NAMES),
        "duplicates_or_rejected": [
            {"feature": "median", "reason": "already present in current handcrafted features"},
            {"feature": "standard_deviation", "reason": "already present in current handcrafted features"},
            {"feature": "variance", "reason": "already present in current handcrafted features"},
            {"feature": "peak_to_peak_amplitude", "reason": "already present as peak_to_peak_amplitude"},
            {"feature": "zero_crossing_rate", "reason": "already present in current handcrafted features"},
        ],
        "final_feature_names": FINAL_FEATURE_NAMES,
        "final_feature_count": len(FINAL_FEATURE_NAMES),
    }
    _save_json(RESULTS_DIR / "feature_audit.json", audit)
    pd.DataFrame(
        [
            {
                "feature_name": name,
                "position": index + 1,
            }
            for index, name in enumerate(FINAL_FEATURE_NAMES)
        ]
    ).to_csv(RESULTS_DIR / "feature_order.csv", index=False)
    return audit


def _save_band_boundaries(params):
    boundaries = {}
    for band_name, band_low, band_high in CANONICAL_BANDS:
        boundaries[band_name] = {
            "canonical_lower": band_low,
            "canonical_upper": band_high,
            "effective_lower": max(band_low, params["f_low"]),
            "effective_upper": min(band_high, params["f_high"]),
        }
    _save_json(RESULTS_DIR / "band_boundaries.json", boundaries)
    return boundaries


def _finalize_payload(method, params, validation_score, validation_metrics, window_metrics, segment_metrics, test_evaluations=1):
    return {
        "experiment": EXP_NAME,
        "method": method,
        "search_evaluations": 1 if method == "Manual SVM" else N_CALLS,
        "validation_score": float(validation_score),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_macro_recall": float(validation_metrics["macro_recall"]),
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_per_class_recall": validation_metrics["per_class_recall"],
        "parameters": {key: float(value) for key, value in params.items()},
        "class_weight": CLASS_WEIGHT,
        "feature_count": len(FINAL_FEATURE_NAMES),
        "primary_evaluation_level": "segment",
        "window_level_test_metrics": _json_metrics(window_metrics),
        "segment_level_test_metrics": _json_metrics(segment_metrics),
        "test_evaluations": int(test_evaluations),
        "canonical_band_boundaries": CANONICAL_BANDS,
    }


def _save_psd_reports(raw_signals, raw_labels, notched_signals, figure_prefix: str):
    before_path = FIGURES_DIR / f"{figure_prefix}_average_psd_before_notch.png"
    after_path = FIGURES_DIR / f"{figure_prefix}_average_psd_after_notch.png"
    zoom_path = FIGURES_DIR / f"{figure_prefix}_average_psd_45_55hz_before_after.png"
    response_path = FIGURES_DIR / f"{figure_prefix}_notch_frequency_response.png"

    _plot_average_psd(raw_signals, raw_labels, before_path, "Average PSD Before 50 Hz Notch")
    _plot_average_psd(notched_signals, raw_labels, after_path, "Average PSD After 50 Hz Notch")
    _plot_psd_zoom(raw_signals, raw_labels, notched_signals, raw_labels, zoom_path)
    _plot_notch_response(response_path)

    notch_summary = _summarise_notch_psd(raw_signals, raw_labels, notched_signals, raw_labels)
    notch_summary.to_csv(RESULTS_DIR / "notch_psd_summary.csv", index=False)
    return before_path, after_path, zoom_path, response_path, notch_summary


def run_experiment():
    _ensure_directories()
    logger = _setup_logging()
    logger.info("Starting %s", EXP_NAME)
    logger.info("Project directory: %s", PROJECT_DIR)
    logger.info("Results directory: %s", RESULTS_DIR)

    feature_audit = _save_feature_audit()
    logger.info("Original feature count: %d", feature_audit["original_feature_count"])
    logger.info("Final feature count: %d", feature_audit["final_feature_count"])

    raw_signals, raw_labels, train_data, test_data, train_ids, test_ids = _load_shared_data()
    raw_signals = [np.asarray(signal, dtype=float) for signal in raw_signals]
    logger.info("Loaded %d raw EEG segments.", len(raw_signals))

    notched_signals = _apply_notch_filter(raw_signals)
    before_path, after_path, zoom_path, response_path, notch_summary = _save_psd_reports(
        raw_signals,
        raw_labels,
        notched_signals,
        EXP_NAME,
    )
    display_image(before_path)
    display_image(after_path)
    display_image(zoom_path)
    display_image(response_path)
    display_dataframe(notch_summary)

    train_signals, train_labels, train_segment_ids = train_data
    test_signals, test_labels, test_segment_ids = test_data

    logger.info("Grouped outer split created before windowing.")
    logger.info("Training segments: %d; held-out test segments: %d", len(train_ids), len(test_ids))
    logger.info("The BO objective receives training segments only.")

    manual_params = {
        "f_low": 0.5,
        "f_high": 60.0,
        "window_size": 5.0,
        "C": 7.671610278174127,
        "gamma": 0.0232081229577128,
    }
    manual_score, manual_metrics, _, _ = _cross_validate_training_segments(
        train_signals,
        train_labels,
        train_segment_ids,
        manual_params,
    )
    manual_window_metrics, manual_segment_metrics, manual_train_shape, manual_test_shape, feature_names, boundaries, train_windows, test_windows, train_validation_report, test_validation_report = _final_test_once(
        train_signals,
        train_labels,
        train_segment_ids,
        test_signals,
        test_labels,
        test_segment_ids,
        manual_params,
    )
    manual_payload = _finalize_payload(
        "Manual SVM",
        manual_params,
        manual_score,
        manual_metrics,
        manual_window_metrics,
        manual_segment_metrics,
    )
    manual_payload["best_iteration"] = 1
    manual_payload["feature_names"] = feature_names
    manual_payload["feature_boundaries"] = boundaries
    manual_payload["train_feature_shape"] = list(manual_train_shape)
    manual_payload["test_feature_shape"] = list(manual_test_shape)
    _save_json(RESULTS_DIR / "experiment_2_manual_final_result.json", manual_payload)
    train_validation_report.to_csv(CACHED_DIR / "manual_train_feature_validation.csv", index=False)
    test_validation_report.to_csv(CACHED_DIR / "manual_test_feature_validation.csv", index=False)
    np.savez_compressed(
        CACHED_DIR / "manual_best_features.npz",
        train_features=np.asarray(train_windows, dtype=object),
        test_features=np.asarray(test_windows, dtype=object),
    )

    logger.info("Manual SVM segment-level validation accuracy: %.4f", manual_metrics["accuracy"])
    logger.info("Manual SVM segment-level test accuracy: %.4f", manual_segment_metrics["accuracy"])

    rng = np.random.default_rng(RANDOM_STATE)
    history = []
    running_path = RESULTS_DIR / "experiment_2_random_search_history_running.csv"

    for iteration in range(1, N_CALLS + 1):
        decoded = _sample_random_params(rng)
        params = _search_space_to_params(
            [
                decoded["f_low_u"],
                decoded["f_high_u"],
                decoded["window_u"],
                decoded["log10_C"],
                decoded["log10_gamma"],
            ]
        )
        score, metrics, feature_shape, _, report, feature_names, boundaries, _, _, _ = _cross_validate_training_segments(
            train_signals,
            train_labels,
            train_segment_ids,
            params,
        )
        row = {
            "iteration": iteration,
            "score": score,
            **decoded,
            **params,
            "class_weight": CLASS_WEIGHT,
            "window_overlap": WINDOW_OVERLAP,
            "cv_folds": CV_FOLDS,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_recall": metrics["macro_recall"],
            "validation_macro_f1": metrics["macro_f1"],
            "feature_rows": int(feature_shape[0]),
            "feature_columns": int(feature_shape[1]),
            "feature_names": json.dumps(feature_names),
            "band_boundaries": json.dumps(boundaries),
            "nan_counts": json.dumps(report["nan_count"].to_list()),
            "inf_counts": json.dumps(report["inf_count"].to_list()),
        }
        for class_name in CLASS_NAMES:
            row[f"validation_recall_{class_name}"] = metrics["per_class_recall"][class_name]
        history.append(row)
        pd.DataFrame(history).to_csv(running_path, index=False)
        logger.info(
            "Random %02d/%02d | score=%.4f | segment_accuracy=%.4f | recall_N=%.4f | recall_F=%.4f",
            iteration,
            N_CALLS,
            score,
            metrics["accuracy"],
            metrics["per_class_recall"]["N"],
            metrics["per_class_recall"]["F"],
        )

    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    best_params = {
        "f_low": float(best_row["f_low"]),
        "f_high": float(best_row["f_high"]),
        "window_size": float(best_row["window_size"]),
        "C": float(best_row["C"]),
        "gamma": float(best_row["gamma"]),
    }
    best_score = float(best_row["score"])
    best_validation_metrics = {
        "accuracy": float(best_row["validation_accuracy"]),
        "macro_recall": float(best_row["validation_macro_recall"]),
        "macro_f1": float(best_row["validation_macro_f1"]),
        "per_class_recall": {
            name: float(best_row[f"validation_recall_{name}"])
            for name in CLASS_NAMES
        },
    }
    random_window_metrics, random_segment_metrics, random_train_shape, random_test_shape, _, boundaries, _, _, random_train_validation_report, random_test_validation_report = _final_test_once(
        train_signals,
        train_labels,
        train_segment_ids,
        test_signals,
        test_labels,
        test_segment_ids,
        best_params,
    )
    random_payload = _finalize_payload(
        "Random Search SVM",
        best_params,
        best_score,
        best_validation_metrics,
        random_window_metrics,
        random_segment_metrics,
    )
    random_payload["best_iteration"] = int(best_row["iteration"])
    random_payload["train_feature_shape"] = list(random_train_shape)
    random_payload["test_feature_shape"] = list(random_test_shape)
    _save_json(RESULTS_DIR / "experiment_2_random_search_final_result.json", random_payload)
    history_df.to_csv(RESULTS_DIR / "experiment_2_random_search_history.csv", index=False)
    random_train_validation_report.to_csv(CACHED_DIR / "random_best_train_feature_validation.csv", index=False)
    random_test_validation_report.to_csv(CACHED_DIR / "random_best_test_feature_validation.csv", index=False)

    def objective(f_low_u, f_high_u, window_u, log10_C, log10_gamma):
        params = _search_space_to_params([f_low_u, f_high_u, window_u, log10_C, log10_gamma])
        score, metrics, feature_shape, _, report, feature_names, boundaries, _, _, _ = _cross_validate_training_segments(
            train_signals,
            train_labels,
            train_segment_ids,
            params,
        )
        row = _flatten_history(
            len(history) + 1,
            {
                **{
                    "f_low_u": f_low_u,
                    "f_high_u": f_high_u,
                    "window_u": window_u,
                    "log10_C": log10_C,
                    "log10_gamma": log10_gamma,
                },
                **params,
            },
            score,
            metrics,
            feature_shape,
            report,
            feature_names,
            boundaries,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(RESULTS_DIR / "experiment_2_bo_history_running.csv", index=False)
        logger.info(
            "BO %02d/%02d | score=%.4f | segment_accuracy=%.4f | recall_N=%.4f | recall_F=%.4f",
            len(history),
            N_CALLS,
            score,
            metrics["accuracy"],
            metrics["per_class_recall"]["N"],
            metrics["per_class_recall"]["F"],
        )
        return -score

    remaining_calls = N_CALLS - len(history)
    if remaining_calls > 0:
        gp_minimize(
            objective,
            SEARCH_SPACE,
            n_calls=remaining_calls,
            n_initial_points=max(0, N_INITIAL_POINTS - len(history)),
            random_state=RANDOM_STATE,
            acq_func="EI",
        )

    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    bo_best_params = {
        "f_low": float(best_row["f_low"]),
        "f_high": float(best_row["f_high"]),
        "window_size": float(best_row["window_size"]),
        "C": float(best_row["C"]),
        "gamma": float(best_row["gamma"]),
    }
    bo_score = float(best_row["score"])
    bo_validation_metrics = {
        "accuracy": float(best_row["validation_accuracy"]),
        "macro_recall": float(best_row["validation_macro_recall"]),
        "macro_f1": float(best_row["validation_macro_f1"]),
        "per_class_recall": {
            name: float(best_row[f"validation_recall_{name}"])
            for name in CLASS_NAMES
        },
    }
    bo_window_metrics, bo_segment_metrics, bo_train_shape, bo_test_shape, bo_feature_names, bo_boundaries, bo_train_windows, bo_test_windows, bo_train_validation_report, bo_test_validation_report = _final_test_once(
        train_signals,
        train_labels,
        train_segment_ids,
        test_signals,
        test_labels,
        test_segment_ids,
        bo_best_params,
    )
    bo_payload = _finalize_payload(
        "BO-SVM",
        bo_best_params,
        bo_score,
        bo_validation_metrics,
        bo_window_metrics,
        bo_segment_metrics,
    )
    bo_payload["best_iteration"] = int(best_row["iteration"])
    bo_payload["best_validation_score"] = float(best_row["score"])
    bo_payload["best_parameters"] = bo_best_params
    bo_payload["feature_names"] = bo_feature_names
    bo_payload["feature_boundaries"] = bo_boundaries
    bo_payload["train_feature_shape"] = list(bo_train_shape)
    bo_payload["test_feature_shape"] = list(bo_test_shape)
    _save_json(RESULTS_DIR / "experiment_2_bo_final_result.json", bo_payload)
    history_df.to_csv(RESULTS_DIR / "experiment_2_bo_history.csv", index=False)
    bo_train_validation_report.to_csv(CACHED_DIR / "bo_best_train_feature_validation.csv", index=False)
    bo_test_validation_report.to_csv(CACHED_DIR / "bo_best_test_feature_validation.csv", index=False)
    np.savez_compressed(
        CACHED_DIR / "bo_best_features.npz",
        train_windows=np.asarray(bo_train_windows, dtype=object),
        test_windows=np.asarray(bo_test_windows, dtype=object),
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "Manual SVM",
                "search_evaluations": 1,
                "validation_score": manual_score,
                "segment_test_accuracy": manual_segment_metrics["accuracy"],
                "segment_test_macro_recall": manual_segment_metrics["macro_recall"],
                "segment_test_macro_f1": manual_segment_metrics["macro_f1"],
                "segment_test_recall_S": manual_segment_metrics["per_class_recall"]["S"],
            },
            {
                "method": "Random Search SVM",
                "search_evaluations": N_CALLS,
                "validation_score": best_score,
                "segment_test_accuracy": random_segment_metrics["accuracy"],
                "segment_test_macro_recall": random_segment_metrics["macro_recall"],
                "segment_test_macro_f1": random_segment_metrics["macro_f1"],
                "segment_test_recall_S": random_segment_metrics["per_class_recall"]["S"],
            },
            {
                "method": "BO-SVM",
                "search_evaluations": N_CALLS,
                "validation_score": bo_score,
                "segment_test_accuracy": bo_segment_metrics["accuracy"],
                "segment_test_macro_recall": bo_segment_metrics["macro_recall"],
                "segment_test_macro_f1": bo_segment_metrics["macro_f1"],
                "segment_test_recall_S": bo_segment_metrics["per_class_recall"]["S"],
            },
        ]
    )
    comparison.to_csv(RESULTS_DIR / "experiment_2_method_comparison.csv", index=False)
    display_dataframe(comparison)

    logger.info("Experiment 2 completed.")
    logger.info("Best BO iteration: %s", bo_payload["best_iteration"])
    logger.info("Best BO parameters: %s", bo_best_params)
    logger.info("Segment-level test accuracy: %.4f", bo_segment_metrics["accuracy"])
    logger.info("Segment-level test macro F1: %.4f", bo_segment_metrics["macro_f1"])
    logger.info("Saved all artifacts under %s", EXP_ROOT)

    print("\nExperiment 2 completed.")
    print("Best BO iteration:", bo_payload["best_iteration"])
    print("Best BO parameters:", bo_best_params)
    print("Segment-level held-out test accuracy:", round(float(bo_segment_metrics["accuracy"]), 4))
    print("Segment-level held-out test macro F1:", round(float(bo_segment_metrics["macro_f1"]), 4))
    print("Held-out test evaluations: 1")

    return {
        "manual": manual_payload,
        "random_search": random_payload,
        "bo": bo_payload,
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_experiment()
