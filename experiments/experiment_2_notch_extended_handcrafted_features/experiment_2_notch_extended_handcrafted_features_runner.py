"""Independent Bonn EEG BO-SVM experiment 2.

raw segment -> fixed 50 Hz notch -> BO-selected Butterworth band-pass ->
windowing -> handcrafted features -> grouped CV objective ->
manual / random / BO comparison with segment-level primary reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import butter, filtfilt, freqz, iirnotch, sosfiltfilt, welch
from scipy.stats import kurtosis, qmc, skew
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_loader import load_bonn_dataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "Dataset"
OLD_SPLIT_MANIFEST = PROJECT_DIR / "results" / "04_grouped_bo_svm" / "grouped_split_manifest.json"

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
CACHED_DIR = EXP_DIR / "cached_features"
LOGS_DIR = EXP_DIR / "logs"

FS = 173.61
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_CALLS = 50
N_INITIAL = 16
CLASS_WEIGHT = "balanced"
WINDOW_OVERLAP = 0.0
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
BUTTER_ORDER = 4
EPS = 1e-12
SOBOL_POOL_SIZE = 8192
EI_XI = 0.0

CLASS_NAMES = ["Z", "O", "N", "F", "S"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

BANDS = [
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

NEW_FEATURE_NAMES = [
    "interquartile_range",
    "mean_absolute_deviation",
    "dominant_frequency",
    "median_frequency",
    "spectral_edge_frequency_90",
    "spectral_centroid",
    "sample_entropy",
    "permutation_entropy",
]

FINAL_FEATURE_NAMES = CURRENT_35_FEATURE_NAMES + NEW_FEATURE_NAMES

SEARCH_BOUNDS = {
    "f_low": (0.5, 2.0),
    "f_high": (40.0, 60.0),
    "window_size": (1.0, 5.0),
    "log10_C": (-1.0, 2.0),
    "log10_gamma": (-4.0, 0.0),
}

MANUAL_CONFIG_PATH = EXP_DIR / "manual_config.json"

RESULT_FILENAMES = {
    "original_feature_list": "original_feature_list.csv",
    "extended_feature_list": "extended_feature_list.csv",
    "feature_summary": "feature_summary.txt",
    "feature_nan_report": "feature_nan_report.csv",
    "feature_correlation": "feature_correlation.csv",
    "grouped_fold_assignments": "grouped_fold_assignments.csv",
    "bo_search_history": "bo_search_history.csv",
    "random_search_history": "random_search_history.csv",
    "bo_best_parameters": "bo_best_parameters.json",
    "random_best_parameters": "random_best_parameters.json",
    "manual_segment_level_metrics": "manual_segment_level_metrics.csv",
    "random_segment_level_metrics": "random_segment_level_metrics.csv",
    "bo_segment_level_metrics": "bo_segment_level_metrics.csv",
    "manual_window_level_metrics": "manual_window_level_metrics.csv",
    "random_window_level_metrics": "random_window_level_metrics.csv",
    "bo_window_level_metrics": "bo_window_level_metrics.csv",
    "method_comparison_segment_level": "method_comparison_segment_level.csv",
    "method_comparison_window_level": "method_comparison_window_level.csv",
    "confusion_matrix_manual_segment_level": "confusion_matrix_manual_segment_level.csv",
    "confusion_matrix_random_segment_level": "confusion_matrix_random_segment_level.csv",
    "confusion_matrix_bo_segment_level": "confusion_matrix_bo_segment_level.csv",
    "selected_parameters_summary": "selected_parameters_summary.png",
    "bo_vs_random_convergence": "bo_vs_random_convergence.png",
    "classwise_recall_comparison_segment_level": "classwise_recall_comparison_segment_level.png",
    "psd_before_notch_all_classes": "psd_before_notch_all_classes.png",
    "psd_after_notch_all_classes": "psd_after_notch_all_classes.png",
    "psd_before_after_45_55hz": "psd_before_after_45_55hz.png",
    "notch_frequency_response": "notch_frequency_response.png",
    "confusion_matrix_manual_segment_level_png": "confusion_matrix_manual_segment_level.png",
    "confusion_matrix_random_segment_level_png": "confusion_matrix_random_segment_level.png",
    "confusion_matrix_bo_segment_level_png": "confusion_matrix_bo_segment_level.png",
}


def ensure_dirs():
    for path in [RESULTS_DIR, FIGURES_DIR, CACHED_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logger():
    logger = logging.getLogger("experiment2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGS_DIR / "experiment_2_notch_extended_handcrafted_features.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(path: Path, payload):
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def load_outer_split():
    if OLD_SPLIT_MANIFEST.exists():
        data = json.loads(OLD_SPLIT_MANIFEST.read_text(encoding="utf-8"))
        return np.asarray(data["train_segment_ids"], dtype=int), np.asarray(data["test_segment_ids"], dtype=int), data
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    segment_ids = np.arange(len(raw_labels), dtype=int)
    train_ids, test_ids = train_test_split(
        segment_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=raw_labels
    )
    manifest = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_segment_ids": np.sort(train_ids).tolist(),
        "test_segment_ids": np.sort(test_ids).tolist(),
    }
    save_json(EXP_DIR / "outer_split_manifest.json", manifest)
    return np.sort(train_ids), np.sort(test_ids), manifest


def save_fold_assignments(train_ids, train_labels):
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_map = {}
    for fold_id, (_, val_idx) in enumerate(splitter.split(np.zeros(len(train_ids)), train_labels, groups=train_ids)):
        for seg_id in np.asarray(train_ids)[val_idx]:
            fold_map[int(seg_id)] = int(fold_id)
    rows = [{"segment_id": int(seg), "label": CLASS_NAMES[int(lbl)], "fold_id": int(fold_map[int(seg)])} for seg, lbl in zip(train_ids, train_labels)]
    fold_df = pd.DataFrame(rows).sort_values("segment_id")
    fold_df.to_csv(CACHED_DIR / "fold_assignments.csv", index=False)
    fold_df.to_csv(RESULTS_DIR / RESULT_FILENAMES["grouped_fold_assignments"], index=False)
    return fold_map


def design_notch_filter():
    return iirnotch(NOTCH_FREQ, NOTCH_Q, FS)


def apply_notch(signals):
    b, a = design_notch_filter()
    return [filtfilt(b, a, np.asarray(sig, dtype=float)) for sig in signals]


def apply_bandpass(signals, f_low, f_high):
    nyquist = FS / 2.0
    sos = butter(BUTTER_ORDER, [f_low / nyquist, f_high / nyquist], btype="bandpass", output="sos")
    return [sosfiltfilt(sos, np.asarray(sig, dtype=float)) for sig in signals]


def segment_signals(signals, labels, segment_ids, window_seconds):
    samples = int(round(float(window_seconds) * FS))
    if samples <= 0:
        raise ValueError("window_size too small")
    step = max(1, int(round(samples * (1.0 - WINDOW_OVERLAP))))
    windows, window_labels, window_groups = [], [], []
    for sig, lbl, seg_id in zip(signals, labels, segment_ids):
        sig = np.asarray(sig, dtype=float)
        for start in range(0, len(sig) - samples + 1, step):
            windows.append(sig[start : start + samples])
            window_labels.append(int(lbl))
            window_groups.append(int(seg_id))
    if not windows:
        raise ValueError("No windows created")
    return np.asarray(windows, dtype=float), np.asarray(window_labels, dtype=int), np.asarray(window_groups, dtype=int)


def safe_entropy_from_hist(values):
    hist, _ = np.histogram(values, bins=32, density=True)
    probs = hist / (np.sum(hist) + EPS)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + EPS))) if len(probs) else 0.0


def spectral_entropy_from_psd(psd):
    probs = np.asarray(psd, dtype=float)
    probs = probs / (np.sum(probs) + EPS)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + EPS))) if len(probs) else 0.0


def approximate_entropy(values, m=2, r_ratio=0.2, max_points=120):
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[idx]
    if len(values) <= m + 2:
        return 0.0
    r = r_ratio * (np.std(values) + EPS)

    def phi(order):
        patterns = np.array([values[i : i + order] for i in range(len(values) - order + 1)])
        if len(patterns) <= 1:
            return 0.0
        counts = []
        for pattern in patterns:
            distances = np.max(np.abs(patterns - pattern), axis=1)
            counts.append(np.mean(distances <= r))
        counts = np.asarray(counts, dtype=float)
        return float(np.mean(np.log(counts + EPS)))

    return float(phi(m) - phi(m + 1))


def sample_entropy(values, m=2, r_ratio=0.2, max_points=120):
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[idx]
    if len(values) <= m + 2:
        return 0.0
    r = r_ratio * (np.std(values) + EPS)

    def phi(order):
        patterns = np.array([values[i : i + order] for i in range(len(values) - order + 1)])
        if len(patterns) <= 1:
            return 0.0
        counts = []
        for pattern in patterns:
            distances = np.max(np.abs(patterns - pattern), axis=1)
            counts.append(np.mean(distances <= r))
        counts = np.asarray(counts, dtype=float)
        return float(np.mean(counts))

    a = phi(m)
    b = phi(m + 1)
    if a <= EPS or b <= EPS:
        return 0.0
    return float(-np.log(b / a))


def permutation_entropy(values, order=3, delay=1):
    values = np.asarray(values, dtype=float)
    if len(values) < order * delay:
        return 0.0
    patterns = []
    for start in range(len(values) - delay * (order - 1)):
        window = values[start : start + delay * order : delay]
        if len(window) == order:
            patterns.append(tuple(np.argsort(window, kind="mergesort")))
    if not patterns:
        return 0.0
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / np.sum(counts)
    return float(-np.sum(probs * np.log2(probs + EPS)))


def band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def band_power_overlap(freqs, psd, band_low, band_high, f_low, f_high):
    low = max(band_low, f_low)
    high = min(band_high, f_high)
    if high <= low:
        return 0.0, (low, high)
    return band_power(freqs, psd, low, high), (low, high)


def extract_window_features(window, f_low, f_high):
    window = np.asarray(window, dtype=float)
    freqs, psd = welch(window, fs=FS, nperseg=min(256, len(window)))

    band_powers = []
    effective_bounds = {}
    band_values = {}
    for band_name, band_low, band_high in BANDS:
        power, bounds = band_power_overlap(freqs, psd, band_low, band_high, f_low, f_high)
        band_powers.append(power)
        band_values[band_name] = power
        effective_bounds[band_name] = bounds

    total_band = float(np.sum(band_powers)) + EPS
    relative = [p / total_band for p in band_powers]
    ratios = [
        band_values["delta"] / (band_values["theta"] + EPS),
        band_values["theta"] / (band_values["alpha"] + EPS),
        band_values["alpha"] / (band_values["beta"] + EPS),
        band_values["beta"] / (band_values["gamma"] + EPS),
        (band_values["delta"] + band_values["theta"]) / (band_values["alpha"] + band_values["beta"] + EPS),
        band_values["gamma"] / total_band,
    ]

    mean = float(np.mean(window))
    std = float(np.std(window))
    var = float(np.var(window))
    maxv = float(np.max(window))
    minv = float(np.min(window))
    med = float(np.median(window))
    ptp = float(np.ptp(window))
    rms = float(np.sqrt(np.mean(window ** 2)))
    line_length = float(np.sum(np.abs(np.diff(window))))
    skewness = float(skew(window, bias=False)) if np.std(window) > EPS else 0.0
    kurt = float(kurtosis(window, bias=False)) if np.std(window) > EPS else 0.0
    energy = float(np.sum(window ** 2))
    zcr = float(np.sum(np.diff(np.signbit(window - mean)) != 0) / max(len(window) - 1, 1))
    shannon = safe_entropy_from_hist(window)
    spectral_entropy = spectral_entropy_from_psd(psd)
    approx = approximate_entropy(window)
    hjorth_activity = var
    first = np.diff(window)
    second = np.diff(first)
    mobility = float(np.sqrt(np.var(first) / (var + EPS)))
    complexity = float(np.sqrt(np.var(second) / (np.var(first) + EPS)) / (mobility + EPS))
    iqr = float(np.percentile(window, 75) - np.percentile(window, 25))
    mad = float(np.mean(np.abs(window - mean)))

    dominant_freq = float(freqs[int(np.argmax(psd))]) if len(psd) else 0.0
    cumulative = np.cumsum(psd)
    median_freq = float(freqs[int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))]) if len(psd) else 0.0
    sef90 = float(freqs[int(np.searchsorted(cumulative, cumulative[-1] * 0.90, side="left"))]) if len(psd) else 0.0
    spectral_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + EPS))
    sampen = sample_entropy(window)
    permen = permutation_entropy(window)

    feature_values = [
        *band_powers,
        *relative,
        *ratios,
        mean,
        std,
        var,
        maxv,
        minv,
        med,
        ptp,
        rms,
        line_length,
        skewness,
        kurt,
        energy,
        zcr,
        shannon,
        spectral_entropy,
        approx,
        hjorth_activity,
        mobility,
        complexity,
        iqr,
        mad,
        dominant_freq,
        median_freq,
        sef90,
        spectral_centroid,
        sampen,
        permen,
    ]
    return feature_values, effective_bounds


def extract_feature_matrix(windows, f_low, f_high):
    rows = []
    bounds = None
    for window in windows:
        values, window_bounds = extract_window_features(window, f_low, f_high)
        rows.append(values)
        if bounds is None:
            bounds = window_bounds
    X = np.asarray(rows, dtype=float)
    nan_counts = np.isnan(X).sum(axis=0).astype(int)
    inf_counts = np.isinf(X).sum(axis=0).astype(int)
    report = pd.DataFrame({"feature_name": FINAL_FEATURE_NAMES, "nan_count": nan_counts, "inf_count": inf_counts})
    if not np.isfinite(X).all():
        raise ValueError("NaN or infinity values found in features")
    return X, report, bounds


def metrics_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(CLASS_NAMES)))
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {name: float(val) for name, val in zip(CLASS_NAMES, per_class_recall)},
        "seizure_sensitivity": float(per_class_recall[CLASS_TO_INDEX["S"]]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def segment_majority_vote(window_pred, window_scores, segment_ids, y_true):
    segment_true, segment_pred = [], []
    for seg_id in np.unique(segment_ids):
        mask = segment_ids == seg_id
        seg_true = np.unique(y_true[mask])
        if len(seg_true) != 1:
            raise RuntimeError("A segment contains multiple true labels")
        counts = np.bincount(window_pred[mask], minlength=len(CLASS_NAMES))
        top = np.flatnonzero(counts == counts.max())
        if len(top) == 1:
            pred = int(top[0])
        else:
            mean_scores = np.mean(window_scores[mask], axis=0)
            pred = int(top[np.argmax(mean_scores[top])])
        segment_true.append(int(seg_true[0]))
        segment_pred.append(pred)
    return np.asarray(segment_true, dtype=int), np.asarray(segment_pred, dtype=int)


def objective_value(segment_metrics):
    return (
        0.35 * segment_metrics["accuracy"]
        + 0.25 * segment_metrics["macro_recall"]
        + 0.20 * segment_metrics["per_class_recall"]["N"]
        + 0.20 * segment_metrics["per_class_recall"]["F"]
    )


def map_unit_to_params(unit_point):
    f_low = SEARCH_BOUNDS["f_low"][0] + unit_point[0] * (SEARCH_BOUNDS["f_low"][1] - SEARCH_BOUNDS["f_low"][0])
    f_high = SEARCH_BOUNDS["f_high"][0] + unit_point[1] * (SEARCH_BOUNDS["f_high"][1] - SEARCH_BOUNDS["f_high"][0])
    window_size = SEARCH_BOUNDS["window_size"][0] + unit_point[2] * (SEARCH_BOUNDS["window_size"][1] - SEARCH_BOUNDS["window_size"][0])
    log10_C = SEARCH_BOUNDS["log10_C"][0] + unit_point[3] * (SEARCH_BOUNDS["log10_C"][1] - SEARCH_BOUNDS["log10_C"][0])
    log10_gamma = SEARCH_BOUNDS["log10_gamma"][0] + unit_point[4] * (SEARCH_BOUNDS["log10_gamma"][1] - SEARCH_BOUNDS["log10_gamma"][0])
    return {
        "f_low": float(f_low),
        "f_high": float(f_high),
        "window_size": float(window_size),
        "log10_C": float(log10_C),
        "log10_gamma": float(log10_gamma),
        "C": float(10 ** log10_C),
        "gamma": float(10 ** log10_gamma),
    }


def unit_key(unit_point):
    return tuple(np.round(np.asarray(unit_point, dtype=float), 12).tolist())


def make_model(params):
    return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], class_weight=CLASS_WEIGHT, decision_function_shape="ovr")


def fit_model_with_gamma(X_train, y_train, params):
    model = make_model(params)
    model.fit(X_train, y_train)
    return model, float(getattr(model, "_gamma", np.nan))


def cv_evaluate(notched_train_signals, train_labels, train_ids, params, fold_map, n_folds=CV_FOLDS):
    bandpassed = apply_bandpass(notched_train_signals, params["f_low"], params["f_high"])
    windows, window_labels, window_groups = segment_signals(bandpassed, train_labels, train_ids, params["window_size"])
    X, feature_report, band_bounds = extract_feature_matrix(windows, params["f_low"], params["f_high"])
    window_fold_ids = np.array([fold_map[int(seg)] for seg in window_groups], dtype=int)

    window_true_all, window_pred_all = [], []
    segment_true_all, segment_pred_all = [], []
    fold_metrics = []
    scaler_reports = []
    effective_gamma_values = []

    inferred_folds = int(np.max(window_fold_ids)) + 1 if len(window_fold_ids) else int(n_folds)
    for fold_id in range(max(int(n_folds), inferred_folds)):
        train_mask = window_fold_ids != fold_id
        val_mask = window_fold_ids == fold_id
        if not np.any(val_mask) or not np.any(train_mask):
            continue
        scaler = StandardScaler().fit(X[train_mask])
        X_train = scaler.transform(X[train_mask])
        X_val = scaler.transform(X[val_mask])
        model, effective_gamma = fit_model_with_gamma(X_train, window_labels[train_mask], params)
        effective_gamma_values.append(effective_gamma)
        val_pred = model.predict(X_val)
        val_scores = model.decision_function(X_val)
        if val_scores.ndim == 1:
            val_scores = np.column_stack([-val_scores, val_scores])
        seg_true, seg_pred = segment_majority_vote(val_pred, val_scores, window_groups[val_mask], window_labels[val_mask])

        window_true_all.extend(window_labels[val_mask].tolist())
        window_pred_all.extend(val_pred.tolist())
        segment_true_all.extend(seg_true.tolist())
        segment_pred_all.extend(seg_pred.tolist())
        fold_metrics.append({
            "fold_id": int(fold_id),
            "train_windows": int(np.sum(train_mask)),
            "validation_windows": int(np.sum(val_mask)),
            "validation_segments": int(len(seg_true)),
            "segment_accuracy": metrics_summary(seg_true, seg_pred)["accuracy"],
        })
        scaler_reports.append({
            "fold_id": int(fold_id),
            "train_windows": int(np.sum(train_mask)),
            "validation_windows": int(np.sum(val_mask)),
        })

    window_metrics = metrics_summary(window_true_all, window_pred_all)
    segment_metrics = metrics_summary(segment_true_all, segment_pred_all)
    objective = objective_value(segment_metrics)
    return {
        "objective": float(objective),
        "window_metrics": window_metrics,
        "segment_metrics": segment_metrics,
        "feature_report": feature_report,
        "feature_count": int(X.shape[1]),
        "feature_shape": list(X.shape),
        "band_boundaries": band_bounds,
        "effective_gamma_folds": [float(v) for v in effective_gamma_values],
        "effective_gamma_mean": float(np.mean(effective_gamma_values)) if effective_gamma_values else float("nan"),
        "fold_metrics": fold_metrics,
        "window_fold_ids": window_fold_ids,
        "scaler_reports": scaler_reports,
    }


def holdout_evaluate(notched_train_signals, train_labels, train_ids, notched_test_signals, test_labels, test_ids, params):
    bandpassed_train = apply_bandpass(notched_train_signals, params["f_low"], params["f_high"])
    train_windows, train_window_labels, train_window_groups = segment_signals(bandpassed_train, train_labels, train_ids, params["window_size"])
    X_train, _, band_bounds = extract_feature_matrix(train_windows, params["f_low"], params["f_high"])

    bandpassed_test = apply_bandpass(notched_test_signals, params["f_low"], params["f_high"])
    test_windows, test_window_labels, test_window_groups = segment_signals(bandpassed_test, test_labels, test_ids, params["window_size"])
    X_test, test_report, _ = extract_feature_matrix(test_windows, params["f_low"], params["f_high"])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model, effective_gamma = fit_model_with_gamma(X_train_s, train_window_labels, params)
    test_pred = model.predict(X_test_s)
    test_scores = model.decision_function(X_test_s)
    if test_scores.ndim == 1:
        test_scores = np.column_stack([-test_scores, test_scores])
    seg_true, seg_pred = segment_majority_vote(test_pred, test_scores, test_window_groups, test_window_labels)
    return {
        "window_metrics": metrics_summary(test_window_labels, test_pred),
        "segment_metrics": metrics_summary(seg_true, seg_pred),
        "feature_report": test_report,
        "feature_shape": list(X_test.shape),
        "band_boundaries": band_bounds,
        "effective_gamma": effective_gamma,
    }


def feature_audit():
    # Proposed additions are kept only if they are not already present in the legacy 35-feature layout.
    duplicate_or_rejected = [
        {"feature": "median", "reason": "already present in current handcrafted features"},
        {"feature": "standard_deviation", "reason": "already present in current handcrafted features"},
        {"feature": "variance", "reason": "already present in current handcrafted features"},
        {"feature": "peak_to_peak_amplitude", "reason": "already present as peak_to_peak_amplitude"},
        {"feature": "zero_crossing_rate", "reason": "already present in current handcrafted features"},
    ]
    payload = {
        "original_feature_names": CURRENT_35_FEATURE_NAMES,
        "original_feature_count": len(CURRENT_35_FEATURE_NAMES),
        "new_feature_names": NEW_FEATURE_NAMES,
        "new_feature_count": len(NEW_FEATURE_NAMES),
        "duplicate_or_rejected": duplicate_or_rejected,
        "final_feature_names": FINAL_FEATURE_NAMES,
        "final_feature_count": len(FINAL_FEATURE_NAMES),
    }
    save_json(RESULTS_DIR / "feature_audit.json", payload)
    pd.DataFrame({"feature_name": CURRENT_35_FEATURE_NAMES, "position": np.arange(1, len(CURRENT_35_FEATURE_NAMES) + 1)}).to_csv(
        RESULTS_DIR / RESULT_FILENAMES["original_feature_list"], index=False
    )
    pd.DataFrame({"feature_name": FINAL_FEATURE_NAMES, "position": np.arange(1, len(FINAL_FEATURE_NAMES) + 1)}).to_csv(
        RESULTS_DIR / RESULT_FILENAMES["extended_feature_list"], index=False
    )
    pd.DataFrame(duplicate_or_rejected).to_csv(RESULTS_DIR / "feature_duplicate_or_rejected.csv", index=False)
    feature_summary_lines = [
        f"Original feature count: {len(CURRENT_35_FEATURE_NAMES)}",
        f"New feature count: {len(NEW_FEATURE_NAMES)}",
        f"Final feature count: {len(FINAL_FEATURE_NAMES)}",
        "",
        "New features:",
        *[f"- {name}" for name in NEW_FEATURE_NAMES],
        "",
        "Duplicate or rejected features:",
        *[f"- {item['feature']}: {item['reason']}" for item in duplicate_or_rejected],
    ]
    (RESULTS_DIR / RESULT_FILENAMES["feature_summary"]).write_text("\n".join(feature_summary_lines) + "\n", encoding="utf-8")
    return payload


def save_notch_psd_reports(raw_signals, raw_labels, notched_signals, output_dir=FIGURES_DIR, summary_dir=RESULTS_DIR):
    def class_average_psd(signals, labels, class_index):
        class_signals = [sig for sig, lbl in zip(signals, labels) if int(lbl) == class_index]
        freqs = None
        psds = []
        for sig in class_signals:
            f, p = welch(sig, fs=FS, nperseg=min(1024, len(sig)), noverlap=512)
            freqs = f if freqs is None else freqs
            psds.append(p)
        return freqs, np.mean(np.asarray(psds), axis=0)

    before_rows, after_rows = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        f0, p0 = class_average_psd(raw_signals, raw_labels, idx)
        f1, p1 = class_average_psd(notched_signals, raw_labels, idx)
        before_rows.append((cls, f0, p0))
        after_rows.append((cls, f1, p1))

    def plot_full(rows, title, path):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cls, freqs, psd in rows:
            ax.plot(freqs, psd, label=cls)
        ax.set_xlim(0, 80)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Average PSD")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Class")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def plot_zoom(before, after, path):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (cls, f0, p0), (_, f1, p1) in zip(before, after):
            mask0 = (f0 >= 45.0) & (f0 <= 55.0)
            mask1 = (f1 >= 45.0) & (f1 <= 55.0)
            ax.plot(f0[mask0], p0[mask0], linestyle="--", label=f"{cls} before")
            ax.plot(f1[mask1], p1[mask1], linewidth=2, label=f"{cls} after")
        ax.axvspan(48.0, 52.0, color="#f59e0b", alpha=0.15, label="48-52 Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Average PSD")
        ax.set_title("Before/After 50 Hz Notch: 45-55 Hz Zoom")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    response_path = output_dir / RESULT_FILENAMES["notch_frequency_response"]
    b, a = design_notch_filter()
    w, h = freqz(b, a, fs=FS, worN=4096)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), EPS)))
    ax.axvline(NOTCH_FREQ, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("50 Hz Notch Frequency Response")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(response_path, dpi=300)
    plt.close(fig)

    before_path = output_dir / RESULT_FILENAMES["psd_before_notch_all_classes"]
    after_path = output_dir / RESULT_FILENAMES["psd_after_notch_all_classes"]
    zoom_path = output_dir / RESULT_FILENAMES["psd_before_after_45_55hz"]
    plot_full(before_rows, "Unfiltered Bonn EEG: Class-wise Average PSD", before_path)
    plot_full(after_rows, "After 50 Hz Notch: Class-wise Average PSD", after_path)
    plot_zoom(before_rows, after_rows, zoom_path)

    rows = []
    for cls, before, after in zip(CLASS_NAMES, before_rows, after_rows):
        freqs0, psd0 = before[1], before[2]
        freqs1, psd1 = after[1], after[2]
        idx0 = int(np.argmin(np.abs(freqs0 - NOTCH_FREQ)))
        idx1 = int(np.argmin(np.abs(freqs1 - NOTCH_FREQ)))
        rows.append({
            "class": cls,
            "psd_50hz_before_mean": float(psd0[idx0]),
            "psd_50hz_after_mean": float(psd1[idx1]),
            "suppression_ratio_after_over_before": float(psd1[idx1] / (psd0[idx0] + EPS)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "notch_psd_summary.csv", index=False)
    return summary


def load_manual_config():
    config = {
        "label": "Fixed conventional RBF-SVM baseline",
        "f_low": 0.5,
        "f_high": 40.0,
        "window_size": 2.0,
        "C": 1.0,
        "gamma": "scale",
        "class_weight": CLASS_WEIGHT,
        "source": "fixed untuned conventional baseline",
    }
    if MANUAL_CONFIG_PATH.exists():
        try:
            existing = json.loads(MANUAL_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing != config:
            save_json(MANUAL_CONFIG_PATH, config)
        return config
    save_json(MANUAL_CONFIG_PATH, config)
    return config


def save_history(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def load_history_if_exists(path):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def save_dataframe(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_confusion_matrix_csv(cm, path):
    df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df.index.name = "true_label"
    save_dataframe(df.reset_index(), path)


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor="white")
    ax.set_facecolor("white")
    cm = np.asarray(cm, dtype=float)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, color="black")
    ax.set_yticklabels(CLASS_NAMES, color="black")
    ax.set_xlabel("Predicted", color="black")
    ax.set_ylabel("True", color="black")
    ax.set_title(title, color="black")
    ax.tick_params(axis="both", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if i == j else "#0b1f3a"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.get_yticklabels(), color="black")
    cbar.outline.set_edgecolor("black")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_metric_tables(method, segment_metrics, window_metrics, output_dir=RESULTS_DIR):
    seg_row = {"method": method, "metric_level": "segment"}
    win_row = {"method": method, "metric_level": "window"}
    for metric_name, value in segment_metrics.items():
        if metric_name == "confusion_matrix":
            continue
        if metric_name == "per_class_recall":
            for cls in CLASS_NAMES:
                seg_row[f"recall_{cls}"] = float(value[cls])
        else:
            seg_row[metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value
    for metric_name, value in window_metrics.items():
        if metric_name == "confusion_matrix":
            continue
        if metric_name == "per_class_recall":
            for cls in CLASS_NAMES:
                win_row[f"recall_{cls}"] = float(value[cls])
        else:
            win_row[metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value
    save_dataframe(pd.DataFrame([seg_row]), output_dir / f"{method.lower()}_segment_level_metrics.csv")
    save_dataframe(pd.DataFrame([win_row]), output_dir / f"{method.lower()}_window_level_metrics.csv")


def save_confusion_outputs(method, segment_metrics, output_dir=RESULTS_DIR, figure_dir=FIGURES_DIR):
    cm = np.asarray(segment_metrics["confusion_matrix"], dtype=int)
    cm_path = output_dir / f"confusion_matrix_{method.lower()}_segment_level.csv"
    save_confusion_matrix_csv(cm, cm_path)
    plot_confusion_matrix(cm, f"{method.title()} Segment-Level Confusion Matrix", figure_dir / f"confusion_matrix_{method.lower()}_segment_level.png")


def save_recall_comparison(method_rows, output_dir=RESULTS_DIR, figure_dir=FIGURES_DIR):
    seg_df = pd.DataFrame(method_rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(seg_df))
    width = 0.15
    for idx, cls in enumerate(CLASS_NAMES):
        ax.bar(x + (idx - 2) * width, seg_df[f"segment_recall_{cls}"], width=width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_df["method"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title("Class-wise Segment-Level Recall Comparison")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / RESULT_FILENAMES["classwise_recall_comparison_segment_level"], dpi=300)
    plt.close(fig)


def save_convergence_plot(bo_history, random_history, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(bo_history["evaluation_number"], bo_history["best_objective_so_far"], marker="o", linewidth=1.5, label="BO")
    ax.plot(random_history["evaluation_number"], random_history["best_objective_so_far"], marker="o", linewidth=1.5, label="Random")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best objective so far")
    ax.set_title("BO vs Random Search Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_selected_parameters_summary(method_rows, output_path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table_rows = []
    for row in method_rows:
        params = row["parameters"]
        table_rows.append([
            row["method"],
            f'{params["f_low"]:.3f}',
            f'{params["f_high"]:.3f}',
            f'{params["window_size"]:.3f}',
            f'{params["C"]:.4g}',
            params["gamma"] if isinstance(params["gamma"], str) else f'{params["gamma"]:.4g}',
            f'{row["best_grouped_cv_objective"]:.4f}',
        ])
    table = ax.table(
        cellText=table_rows,
        colLabels=["Method", "f_low", "f_high", "window", "C", "gamma", "Best obj"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def metric_row_from_metrics(method, metrics, level):
    row = {"method": method, "metric_level": level}
    row["accuracy"] = float(metrics["accuracy"])
    row["macro_recall"] = float(metrics["macro_recall"])
    row["macro_f1"] = float(metrics["macro_f1"])
    row["seizure_sensitivity"] = float(metrics["seizure_sensitivity"])
    for cls in CLASS_NAMES:
        row[f"recall_{cls}"] = float(metrics["per_class_recall"][cls])
    return row


def save_feature_quality_artifacts(train_signals, train_labels, train_ids, params, output_dir=RESULTS_DIR):
    notched = apply_notch(train_signals)
    bandpassed = apply_bandpass(notched, params["f_low"], params["f_high"])
    windows, _, _ = segment_signals(bandpassed, train_labels, train_ids, params["window_size"])
    X, feature_report, _ = extract_feature_matrix(windows, params["f_low"], params["f_high"])
    feature_report.rename(columns={"nan_count": "nan_count", "inf_count": "inf_count"}).to_csv(
        output_dir / RESULT_FILENAMES["feature_nan_report"], index=False
    )
    corr_values = np.corrcoef(X, rowvar=False)
    corr_values = np.nan_to_num(corr_values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_values, 1.0)
    corr = pd.DataFrame(corr_values, index=FINAL_FEATURE_NAMES, columns=FINAL_FEATURE_NAMES)
    corr.to_csv(output_dir / RESULT_FILENAMES["feature_correlation"])
    return {
        "feature_count": int(X.shape[1]),
        "feature_report": feature_report,
        "correlation_shape": list(corr.shape),
    }


def fit_gp(x_obs, y_obs):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(5),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=RANDOM_STATE,
    )
    gp.fit(np.asarray(x_obs, dtype=float), np.asarray(y_obs, dtype=float))
    return gp


def ei_scores(gp, x_candidates, best_y):
    mu, std = gp.predict(np.asarray(x_candidates, dtype=float), return_std=True)
    std = np.maximum(std, EPS)
    imp = mu - best_y - EI_XI
    z = imp / std
    from scipy.stats import norm

    return imp * norm.cdf(z) + std * norm.pdf(z)


def unit_to_actual(unit):
    return {
        "f_low": SEARCH_BOUNDS["f_low"][0] + unit[0] * (SEARCH_BOUNDS["f_low"][1] - SEARCH_BOUNDS["f_low"][0]),
        "f_high": SEARCH_BOUNDS["f_high"][0] + unit[1] * (SEARCH_BOUNDS["f_high"][1] - SEARCH_BOUNDS["f_high"][0]),
        "window_size": SEARCH_BOUNDS["window_size"][0] + unit[2] * (SEARCH_BOUNDS["window_size"][1] - SEARCH_BOUNDS["window_size"][0]),
        "log10_C": SEARCH_BOUNDS["log10_C"][0] + unit[3] * (SEARCH_BOUNDS["log10_C"][1] - SEARCH_BOUNDS["log10_C"][0]),
        "log10_gamma": SEARCH_BOUNDS["log10_gamma"][0] + unit[4] * (SEARCH_BOUNDS["log10_gamma"][1] - SEARCH_BOUNDS["log10_gamma"][0]),
    }


def actual_from_unit(unit):
    params = unit_to_actual(unit)
    return {
        **params,
        "C": float(10 ** params["log10_C"]),
        "gamma": float(10 ** params["log10_gamma"]),
    }


def eval_record(eval_no, phase, unit, params, result, elapsed, best_so_far, duplicate_hits=0):
    seg = result["segment_metrics"]
    win = result["window_metrics"]
    row = {
        "evaluation_number": int(eval_no),
        "phase": phase,
        "duplicate_hits": int(duplicate_hits),
        "f_low_unit": float(unit[0]),
        "f_high_unit": float(unit[1]),
        "window_unit": float(unit[2]),
        "log10_C": float(unit[3]),
        "log10_gamma": float(unit[4]),
        "f_low": float(params["f_low"]),
        "f_high": float(params["f_high"]),
        "window_size": float(params["window_size"]),
        "C": float(params["C"]),
        "gamma": float(params["gamma"]),
        "objective": float(result["objective"]),
        "best_objective_so_far": float(best_so_far),
        "elapsed_seconds": float(elapsed),
        "segment_accuracy": float(seg["accuracy"]),
        "segment_macro_recall": float(seg["macro_recall"]),
        "segment_macro_f1": float(seg["macro_f1"]),
        "segment_seizure_sensitivity": float(seg["seizure_sensitivity"]),
        "window_accuracy": float(win["accuracy"]),
        "window_macro_recall": float(win["macro_recall"]),
        "window_macro_f1": float(win["macro_f1"]),
        "window_seizure_sensitivity": float(win["seizure_sensitivity"]),
        "feature_count": int(result["feature_count"]),
        "feature_shape": json.dumps(result["feature_shape"]),
        "band_boundaries": json.dumps(result["band_boundaries"]),
        "fold_metrics": json.dumps(result["fold_metrics"]),
        "nan_counts": json.dumps(result["feature_report"]["nan_count"].astype(int).tolist()),
        "inf_counts": json.dumps(result["feature_report"]["inf_count"].astype(int).tolist()),
    }
    for cls in CLASS_NAMES:
        row[f"segment_recall_{cls}"] = float(seg["per_class_recall"][cls])
        row[f"window_recall_{cls}"] = float(win["per_class_recall"][cls])
    return row


def evaluate_search(method_name, train_signals, train_labels, train_ids, fold_map, unit_candidates, logger, pool=None):
    notched_train = apply_notch(train_signals)
    history = []
    observed_units = []
    observed_objectives = []
    seen = set()
    start = time.perf_counter()

    for idx, unit in enumerate(unit_candidates, start=1):
        key = unit_key(unit)
        duplicate_hits = 0
        if key in seen:
            duplicate_hits += 1
        seen.add(key)
        params = actual_from_unit(unit)
        result = cv_evaluate(notched_train, train_labels, train_ids, params, fold_map)
        observed_units.append(np.asarray(unit, dtype=float))
        observed_objectives.append(result["objective"])
        best_so_far = max(observed_objectives)
        elapsed = time.perf_counter() - start
        history.append(eval_record(idx, "initial" if idx <= N_INITIAL and method_name == "BO-SVM" else method_name.lower(), unit, params, result, elapsed, best_so_far, duplicate_hits))
        save_history(history, RESULTS_DIR / f"{method_name.lower().replace(' ', '_')}_history_running.csv")
        logger.info(
            "%s %02d/%02d | obj=%.4f | seg_acc=%.4f | seg_macro_recall=%.4f | seg_recall_N=%.4f | seg_recall_F=%.4f",
            method_name,
            idx,
            len(unit_candidates),
            result["objective"],
            result["segment_metrics"]["accuracy"],
            result["segment_metrics"]["macro_recall"],
            result["segment_metrics"]["per_class_recall"]["N"],
            result["segment_metrics"]["per_class_recall"]["F"],
        )
    return history, np.asarray(observed_units), np.asarray(observed_objectives)


def unique_random_unit_candidate(rng, seen):
    while True:
        unit = np.array([
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
        ])
        key = unit_key(unit)
        if key not in seen:
            seen.add(key)
            return unit


def choose_next_bo_candidate(gp, seen, pool, best_y):
    remaining = np.array([p for p in pool if unit_key(p) not in seen], dtype=float)
    if len(remaining) == 0:
        raise RuntimeError("No candidates left for EI selection")
    scores = ei_scores(gp, remaining, best_y)
    order = np.argsort(scores)[::-1]
    for idx in order:
        unit = remaining[idx]
        key = unit_key(unit)
        if key not in seen:
            seen.add(key)
            return unit, float(scores[idx])
    unit = remaining[order[0]]
    return unit, float(scores[order[0]])


def run_bo(train_signals, train_labels, train_ids, fold_map, logger, resume=False, start_eval=None):
    sobol_init = qmc.Sobol(d=5, scramble=True, seed=RANDOM_STATE).random_base2(m=4)
    sobol_pool = qmc.Sobol(d=5, scramble=True, seed=RANDOM_STATE + 1).random_base2(m=13)
    running_path = RESULTS_DIR / "bo_search_history_running.csv"
    history_df = load_history_if_exists(running_path) if resume else pd.DataFrame()
    history = history_df.to_dict("records") if len(history_df) else []
    observed_units = []
    observed_objectives = []
    seen = set()
    start = time.perf_counter()
    notched_train = apply_notch(train_signals)

    if len(history_df):
        for _, row in history_df.iterrows():
            unit = np.array([
                float(row["f_low_unit"]),
                float(row["f_high_unit"]),
                float(row["window_unit"]),
                float(row["log10_C"]),
                float(row["log10_gamma"]),
            ])
            observed_units.append(unit)
            observed_objectives.append(float(row["objective"]))
            seen.add(unit_key(unit))

    logger.info(
        "BO settings: GP surrogate=sklearn GaussianProcessRegressor, Mat\'ern nu=2.5, ARD=True, alpha=1e-8, white_noise=1e-6, seed=%d",
        RANDOM_STATE,
    )
    logger.info("Acquisition: Expected Improvement, optimisation by argmax over a Sobol candidate pool of %d points", SOBOL_POOL_SIZE)

    if start_eval is None:
        start_eval = len(history) + 1
    else:
        start_eval = int(start_eval)
        if resume and len(history) < start_eval - 1:
            raise ValueError(
                f"Cannot start at evaluation {start_eval}: history has only {len(history)} completed evaluations."
            )
    if start_eval > N_CALLS:
        logger.info("BO history already contains %d evaluations; nothing to resume.", len(history))
    for eval_no in range(start_eval, N_CALLS + 1):
        if eval_no <= N_INITIAL:
            unit = np.asarray(sobol_init[eval_no - 1], dtype=float)
            phase = "initial_design"
        else:
            gp = fit_gp(observed_units, observed_objectives)
            unit, _ = choose_next_bo_candidate(gp, seen, sobol_pool, float(np.max(observed_objectives)))
            phase = "ei"
        key = unit_key(unit)
        seen.add(key)
        params = actual_from_unit(unit)
        result = cv_evaluate(notched_train, train_labels, train_ids, params, fold_map)
        observed_units.append(unit)
        observed_objectives.append(result["objective"])
        best_so_far = float(np.max(observed_objectives))
        elapsed = time.perf_counter() - start
        row = eval_record(eval_no, phase, unit, params, result, elapsed, best_so_far)
        history.append(row)
        save_history(history, running_path)
        logger.info(
            "BO %02d/%02d | phase=%s | obj=%.4f | best=%.4f | seg_acc=%.4f | seg_recall_N=%.4f | seg_recall_F=%.4f",
            eval_no,
            N_CALLS,
            phase,
            result["objective"],
            best_so_far,
            result["segment_metrics"]["accuracy"],
            result["segment_metrics"]["per_class_recall"]["N"],
            result["segment_metrics"]["per_class_recall"]["F"],
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / RESULT_FILENAMES["bo_search_history"], index=False)
    best_row = history_df.loc[history_df["objective"].idxmax()].to_dict()
    best_params = {
        "f_low": float(best_row["f_low"]),
        "f_high": float(best_row["f_high"]),
        "window_size": float(best_row["window_size"]),
        "C": float(best_row["C"]),
        "gamma": float(best_row["gamma"]),
    }
    metadata = {
        "matern_nu": 2.5,
        "ard_length_scales": True,
        "gp_alpha": 1e-8,
        "white_noise_level": 1e-6,
        "random_seed": RANDOM_STATE,
        "acquisition": "Expected Improvement",
        "acquisition_optimization": f"argmax over Sobol pool of {SOBOL_POOL_SIZE} candidates",
        "initial_design": "scrambled Sobol",
        "initial_design_points": N_INITIAL,
        "total_evaluations": N_CALLS,
        "fitted_kernel": str(fit_gp(observed_units, observed_objectives).kernel_),
        "best_row": best_row,
    }
    save_json(RESULTS_DIR / "experiment_2_bo_metadata.json", metadata)
    save_json(RESULTS_DIR / RESULT_FILENAMES["bo_best_parameters"], best_params)
    return history_df, best_params, best_row


def run_random_search(train_signals, train_labels, train_ids, fold_map, logger):
    rng = np.random.default_rng(RANDOM_STATE)
    seen = set()
    unit_candidates = [unique_random_unit_candidate(rng, seen) for _ in range(N_CALLS)]
    notched_train = apply_notch(train_signals)
    history = []
    start = time.perf_counter()
    for eval_no, unit in enumerate(unit_candidates, start=1):
        params = actual_from_unit(unit)
        result = cv_evaluate(notched_train, train_labels, train_ids, params, fold_map)
        elapsed = time.perf_counter() - start
        history.append(eval_record(eval_no, "random", unit, params, result, elapsed, max([r["objective"] for r in history] + [result["objective"]])))
        save_history(history, RESULTS_DIR / "random_search_history_running.csv")
        logger.info(
            "Random %02d/%02d | obj=%.4f | best=%.4f | seg_acc=%.4f | seg_recall_N=%.4f | seg_recall_F=%.4f",
            eval_no,
            N_CALLS,
            result["objective"],
            max([r["objective"] for r in history]),
            result["segment_metrics"]["accuracy"],
            result["segment_metrics"]["per_class_recall"]["N"],
            result["segment_metrics"]["per_class_recall"]["F"],
        )
    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / RESULT_FILENAMES["random_search_history"], index=False)
    best_row = history_df.loc[history_df["objective"].idxmax()].to_dict()
    best_params = {
        "f_low": float(best_row["f_low"]),
        "f_high": float(best_row["f_high"]),
        "window_size": float(best_row["window_size"]),
        "C": float(best_row["C"]),
        "gamma": float(best_row["gamma"]),
    }
    save_json(RESULTS_DIR / RESULT_FILENAMES["random_best_parameters"], best_params)
    return history_df, best_params, best_row


def run_manual(train_signals, train_labels, train_ids, fold_map):
    config = load_manual_config()
    notched_train = apply_notch(train_signals)
    params = {k: config[k] for k in ["f_low", "f_high", "window_size", "C", "gamma"]}
    cv_result = cv_evaluate(notched_train, train_labels, train_ids, params, fold_map)
    return config, cv_result, params


def write_metric_artifacts(method_label, method_key, cv_result, holdout_result, best_params, best_grouped_cv_objective, output_dir=RESULTS_DIR):
    seg_test = holdout_result["segment_metrics"]
    win_test = holdout_result["window_metrics"]

    seg_row = metric_row_from_metrics(method_label, seg_test, "segment")
    win_row = metric_row_from_metrics(method_label, win_test, "window")
    if "effective_gamma" in holdout_result:
        seg_row["effective_gamma"] = float(holdout_result["effective_gamma"])
        win_row["effective_gamma"] = float(holdout_result["effective_gamma"])
    save_dataframe(pd.DataFrame([seg_row]), output_dir / RESULT_FILENAMES[f"{method_key}_segment_level_metrics"])
    save_dataframe(pd.DataFrame([win_row]), output_dir / RESULT_FILENAMES[f"{method_key}_window_level_metrics"])
    save_confusion_outputs(method_key, seg_test, output_dir=output_dir, figure_dir=FIGURES_DIR)
    best_params = dict(best_params)
    if "effective_gamma_mean" in cv_result:
        best_params["effective_gamma_cv_folds"] = cv_result.get("effective_gamma_folds", [])
        best_params["effective_gamma_cv_mean"] = float(cv_result["effective_gamma_mean"])
    if "effective_gamma" in holdout_result:
        best_params["effective_gamma_holdout"] = float(holdout_result["effective_gamma"])
    save_json(output_dir / f"{method_key}_best_parameters.json", best_params)

    return {
        "method": method_label,
        "parameters": best_params,
        "best_grouped_cv_objective": float(best_grouped_cv_objective),
        "cv_segment_metrics": cv_result["segment_metrics"],
        "cv_window_metrics": cv_result["window_metrics"],
        "segment_test_metrics": seg_test,
        "window_test_metrics": win_test,
        "search_evaluations": None,
    }


def save_method_comparison(method_payloads, output_dir=RESULTS_DIR):
    segment_rows = []
    window_rows = []
    for payload in method_payloads:
        seg = payload["segment_test_metrics"]
        win = payload["window_test_metrics"]
        segment_rows.append({
            "method": payload["method"],
            "best_grouped_cv_objective": float(payload["best_grouped_cv_objective"]),
            "segment_accuracy": seg["accuracy"],
            "segment_macro_recall": seg["macro_recall"],
            "segment_recall_Z": seg["per_class_recall"]["Z"],
            "segment_recall_O": seg["per_class_recall"]["O"],
            "segment_recall_N": seg["per_class_recall"]["N"],
            "segment_recall_F": seg["per_class_recall"]["F"],
            "segment_recall_S": seg["per_class_recall"]["S"],
            "segment_seizure_sensitivity": seg["seizure_sensitivity"],
            "best_parameters": json.dumps(payload["parameters"]),
        })
        window_rows.append({
            "method": payload["method"],
            "window_accuracy": win["accuracy"],
            "window_macro_recall": win["macro_recall"],
            "window_recall_Z": win["per_class_recall"]["Z"],
            "window_recall_O": win["per_class_recall"]["O"],
            "window_recall_N": win["per_class_recall"]["N"],
            "window_recall_F": win["per_class_recall"]["F"],
            "window_recall_S": win["per_class_recall"]["S"],
            "window_seizure_sensitivity": win["seizure_sensitivity"],
            "best_parameters": json.dumps(payload["parameters"]),
        })
    save_dataframe(pd.DataFrame(segment_rows), output_dir / RESULT_FILENAMES["method_comparison_segment_level"])
    save_dataframe(pd.DataFrame(window_rows), output_dir / RESULT_FILENAMES["method_comparison_window_level"])
    return pd.DataFrame(segment_rows), pd.DataFrame(window_rows)


def smoke_test():
    ensure_dirs()
    smoke_root = EXP_DIR / "smoke_test"
    smoke_root.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    smoke_results = smoke_root / f"run_{stamp}" / "results"
    smoke_figures = smoke_root / f"run_{stamp}" / "figures"
    smoke_logs = smoke_root / f"run_{stamp}" / "logs"
    for path in [smoke_results, smoke_figures, smoke_logs]:
        path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger()
    logger.info("Starting smoke test for experiment_2_notch_extended_handcrafted_features")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids, _ = load_outer_split()
    train_signals = [raw_signals[i] for i in train_ids]
    train_labels = raw_labels[train_ids]

    subset_ids = []
    for cls in range(len(CLASS_NAMES)):
        cls_ids = [int(seg) for seg, lbl in zip(train_ids, train_labels) if int(lbl) == cls][:10]
        subset_ids.extend(cls_ids)
    subset_ids = np.asarray(sorted(subset_ids), dtype=int)
    subset_mask = np.isin(train_ids, subset_ids)
    smoke_train_ids = train_ids[subset_mask]
    smoke_train_signals = [raw_signals[i] for i in smoke_train_ids]
    smoke_train_labels = raw_labels[smoke_train_ids]
    smoke_splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    smoke_fold_map = {}
    smoke_fold_rows = []
    for fold_id, (_, val_idx) in enumerate(smoke_splitter.split(np.zeros(len(smoke_train_ids)), smoke_train_labels, groups=smoke_train_ids)):
        for seg_id in np.asarray(smoke_train_ids)[val_idx]:
            smoke_fold_map[int(seg_id)] = int(fold_id)
    for seg, lbl in zip(smoke_train_ids, smoke_train_labels):
        smoke_fold_rows.append({"segment_id": int(seg), "label": CLASS_NAMES[int(lbl)], "fold_id": int(smoke_fold_map[int(seg)])})
    pd.DataFrame(smoke_fold_rows).sort_values("segment_id").to_csv(smoke_results / "grouped_fold_assignments.csv", index=False)

    notched_before = apply_notch(raw_signals)
    psd_before = save_notch_psd_reports(raw_signals, raw_labels, notched_before, output_dir=smoke_figures, summary_dir=smoke_results)
    reduction_ok = bool((psd_before["psd_50hz_after_mean"] < psd_before["psd_50hz_before_mean"]).all())

    smoke_config = {"f_low": 0.5, "f_high": 40.0, "window_size": 2.0, "C": 1.0, "gamma": "scale"}
    feat_check = save_feature_quality_artifacts(smoke_train_signals, smoke_train_labels, smoke_train_ids, smoke_config, output_dir=smoke_results)
    candidate_units = [
        np.array([0.1, 0.8, 0.3, 0.5, 0.2]),
        np.array([0.7, 0.2, 0.6, 0.4, 0.7]),
        np.array([0.4, 0.6, 0.1, 0.8, 0.3]),
    ]
    notched_subset = apply_notch(smoke_train_signals)
    candidate_features = []
    validation_ok = True
    scaler_training_only_ok = True
    majority_vote_ok = True
    leakage_ok = True
    for unit in candidate_units:
        params = actual_from_unit(unit)
        params["f_low"] = smoke_config["f_low"]
        params["f_high"] = smoke_config["f_high"]
        params["window_size"] = smoke_config["window_size"]
        result = cv_evaluate(notched_subset, smoke_train_labels, smoke_train_ids, params, smoke_fold_map, n_folds=2)
        candidate_features.append(result["feature_count"])
        validation_ok &= np.isfinite(result["segment_metrics"]["accuracy"])
        validation_ok &= np.isfinite(result["window_metrics"]["accuracy"])
        leakage_ok &= all(r["train_windows"] > 0 and r["validation_windows"] > 0 for r in result["fold_metrics"])
        majority_vote_ok &= all(r["validation_segments"] > 0 for r in result["fold_metrics"])
        scaler_training_only_ok &= all(r["train_windows"] > 0 for r in result["scaler_reports"])
    feat_const_ok = len(set(candidate_features)) == 1

    smoke_payload = {
        "notch_reduction_ok": reduction_ok,
        "feature_count_constant": feat_const_ok,
        "no_nan_inf": int(feat_check["feature_report"]["nan_count"].sum()) == 0 and int(feat_check["feature_report"]["inf_count"].sum()) == 0,
        "group_leakage_ok": leakage_ok,
        "scaler_training_only_ok": scaler_training_only_ok,
        "majority_vote_ok": majority_vote_ok,
        "validation_values_finite": validation_ok,
        "feature_count": int(feat_check["feature_count"]),
        "original_feature_count": len(CURRENT_35_FEATURE_NAMES),
        "final_feature_count": len(FINAL_FEATURE_NAMES),
        "estimated_full_run_evaluations": N_CALLS,
        "estimated_full_run_multiplier": 50 / 3,
        "output_dir": str(smoke_results.parent),
    }
    save_json(smoke_results / "smoke_test_summary.json", smoke_payload)
    logger.info("Smoke test summary: %s", smoke_payload)
    print(pd.DataFrame([smoke_payload]).to_string(index=False))
    return smoke_payload


def run_full_experiment(resume=False, start_eval=None):
    ensure_dirs()
    logger = setup_logger()
    start = time.perf_counter()
    logger.info("Starting experiment_2_notch_extended_handcrafted_features")
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", platform.platform())
    logger.info("NumPy: %s | pandas: %s | SciPy: %s | scikit-learn: %s", np.__version__, pd.__version__, scipy.__version__, sklearn.__version__)
    logger.info("Seeds: random_state=%d", RANDOM_STATE)

    feature_audit_payload = feature_audit()
    logger.info("Original feature count: %d", feature_audit_payload["original_feature_count"])
    logger.info("Final feature count: %d", feature_audit_payload["final_feature_count"])

    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids, outer_manifest = load_outer_split()
    train_signals = [raw_signals[i] for i in train_ids]
    train_labels = raw_labels[train_ids]
    test_signals = [raw_signals[i] for i in test_ids]
    test_labels = raw_labels[test_ids]
    fold_map = save_fold_assignments(train_ids, train_labels)
    save_json(EXP_DIR / "outer_split_manifest_used.json", outer_manifest)

    save_feature_quality_artifacts(train_signals, train_labels, train_ids, load_manual_config())

    notched_all = apply_notch(raw_signals)
    notch_summary = save_notch_psd_reports(raw_signals, raw_labels, notched_all)
    logger.info("Saved notch PSD reports and frequency response")
    logger.info("\n%s", notch_summary.to_string(index=False))

    method_payloads = []

    bo_history, bo_best_params, bo_best_row = run_bo(
        train_signals, train_labels, train_ids, fold_map, logger, resume=resume, start_eval=start_eval
    )
    bo_cv_result = cv_evaluate(apply_notch(train_signals), train_labels, train_ids, bo_best_params, fold_map)
    bo_holdout = holdout_evaluate(apply_notch(train_signals), train_labels, train_ids, apply_notch(test_signals), test_labels, test_ids, bo_best_params)
    bo_payload = write_metric_artifacts("BO", "bo", bo_cv_result, bo_holdout, bo_best_params, bo_history["objective"].max(), output_dir=RESULTS_DIR)
    bo_payload["search_evaluations"] = int(len(bo_history))
    method_payloads.append(bo_payload)

    random_history, random_best_params, random_best_row = run_random_search(train_signals, train_labels, train_ids, fold_map, logger)
    random_cv_result = cv_evaluate(apply_notch(train_signals), train_labels, train_ids, random_best_params, fold_map)
    random_holdout = holdout_evaluate(apply_notch(train_signals), train_labels, train_ids, apply_notch(test_signals), test_labels, test_ids, random_best_params)
    random_payload = write_metric_artifacts("Random Search", "random", random_cv_result, random_holdout, random_best_params, random_history["objective"].max(), output_dir=RESULTS_DIR)
    random_payload["search_evaluations"] = int(len(random_history))
    method_payloads.append(random_payload)

    manual_config, manual_cv_result, manual_params = run_manual(train_signals, train_labels, train_ids, fold_map)
    manual_holdout = holdout_evaluate(apply_notch(train_signals), train_labels, train_ids, apply_notch(test_signals), test_labels, test_ids, manual_params)
    manual_payload = write_metric_artifacts("Fixed conventional RBF-SVM baseline", "manual", manual_cv_result, manual_holdout, manual_params, manual_cv_result["objective"], output_dir=RESULTS_DIR)
    manual_payload["search_evaluations"] = 1
    manual_payload["manual_config"] = manual_config
    method_payloads.append(manual_payload)

    save_method_comparison(method_payloads)
    save_recall_comparison([
        {
            "method": payload["method"],
            **{f"segment_recall_{cls}": payload["segment_test_metrics"]["per_class_recall"][cls] for cls in CLASS_NAMES},
        }
        for payload in method_payloads
    ])
    save_convergence_plot(bo_history, random_history, RESULTS_DIR / RESULT_FILENAMES["bo_vs_random_convergence"])
    save_selected_parameters_summary(method_payloads, FIGURES_DIR / RESULT_FILENAMES["selected_parameters_summary"])

    total_runtime = time.perf_counter() - start
    summary_rows = []
    for payload in method_payloads:
        seg = payload["segment_test_metrics"]
        win = payload["window_test_metrics"]
        summary_rows.append({
            "method": payload["method"],
            "best_grouped_cv_objective": payload["best_grouped_cv_objective"],
            "segment_accuracy": seg["accuracy"],
            "segment_macro_recall": seg["macro_recall"],
            "segment_recall_Z": seg["per_class_recall"]["Z"],
            "segment_recall_O": seg["per_class_recall"]["O"],
            "segment_recall_N": seg["per_class_recall"]["N"],
            "segment_recall_F": seg["per_class_recall"]["F"],
            "segment_recall_S": seg["per_class_recall"]["S"],
            "segment_seizure_sensitivity": seg["seizure_sensitivity"],
            "window_accuracy": win["accuracy"],
            "window_macro_recall": win["macro_recall"],
            "window_recall_Z": win["per_class_recall"]["Z"],
            "window_recall_O": win["per_class_recall"]["O"],
            "window_recall_N": win["per_class_recall"]["N"],
            "window_recall_F": win["per_class_recall"]["F"],
            "window_recall_S": win["per_class_recall"]["S"],
            "window_seizure_sensitivity": win["seizure_sensitivity"],
            "parameters": json.dumps(payload["parameters"]),
        })
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "final_summary_table.csv", index=False)
    logger.info("Total runtime: %.2f s", total_runtime)
    logger.info("F-to-N confusion counts: %s", {
        payload["method"]: int(payload["segment_test_metrics"]["confusion_matrix"][CLASS_TO_INDEX["F"]][CLASS_TO_INDEX["N"]])
        for payload in method_payloads
    })
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return {
        "summary": summary_rows,
        "runtime_seconds": total_runtime,
        "method_payloads": method_payloads,
        "bo_history": bo_history,
        "random_history": random_history,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true", help="Resume BO from bo_search_history_running.csv if present")
    parser.add_argument("--start-eval", type=int, default=None, help="Start BO from this evaluation number")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "smoke":
        smoke_test()
    else:
        run_full_experiment(resume=args.resume or args.start_eval is not None, start_eval=args.start_eval)
