"""
objective.py

Black-box objective function for EEG-SVM parameter tuning.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features_by_setting
from metrics import classification_metrics_summary
from model import train_validate_test_svm
from preprocessing import bandpass_filter_signals, segment_windows


CLASS_NAMES = ["Z", "O", "N", "F", "S"]


def _empty_metrics():
    """Return an empty metrics dictionary for failed evaluations."""
    return {
        "accuracy": 0.0,
        "macro_recall": 0.0,
        "per_class_recall": {class_name: 0.0 for class_name in CLASS_NAMES},
        "seizure_sensitivity": 0.0,
        "confusion_matrix": np.zeros((5, 5), dtype=int),
    }


def _normalise_class_weight(class_weight):
    """Convert BO-friendly class_weight names into sklearn values."""
    if class_weight in (None, "none", "None"):
        return None
    if class_weight == "balanced":
        return "balanced"
    raise ValueError(f"Unknown class_weight: {class_weight}")


def _score_from_metrics(metrics, score_mode):
    """Compute the scalar optimization score from a metrics dictionary."""
    accuracy = metrics["accuracy"]
    macro_recall = metrics["macro_recall"]
    recall_n = metrics["per_class_recall"]["N"]
    recall_f = metrics["per_class_recall"]["F"]

    if score_mode == "accuracy":
        return float(accuracy)
    if score_mode == "accuracy_macro_recall":
        return float(0.7 * accuracy + 0.3 * macro_recall)
    if score_mode == "nf_focused":
        return float(
            0.35 * accuracy
            + 0.25 * macro_recall
            + 0.20 * recall_n
            + 0.20 * recall_f
        )
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _average_metrics(fold_metrics):
    """Average scalar metrics across CV folds and sum confusion matrices."""
    if not fold_metrics:
        return _empty_metrics()

    return {
        "accuracy": float(np.mean([metrics["accuracy"] for metrics in fold_metrics])),
        "macro_recall": float(
            np.mean([metrics["macro_recall"] for metrics in fold_metrics])
        ),
        "per_class_recall": {
            class_name: float(
                np.mean(
                    [
                        metrics["per_class_recall"][class_name]
                        for metrics in fold_metrics
                    ]
                )
            )
            for class_name in CLASS_NAMES
        },
        "seizure_sensitivity": float(
            np.mean([metrics["seizure_sensitivity"] for metrics in fold_metrics])
        ),
        "confusion_matrix": np.sum(
            [metrics["confusion_matrix"] for metrics in fold_metrics],
            axis=0,
        ),
    }


def _cross_validate_svm(
    X,
    y,
    C,
    gamma,
    class_weight,
    score_mode,
    cv_folds,
    random_state=42,
):
    """
    Evaluate one parameter set with stratified K-fold CV.

    The scaler is fitted inside each training fold only, so validation folds do
    not leak into feature normalization.
    """
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_metrics = []
    fold_scores = []

    for train_index, validation_index in splitter.split(X, y):
        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)

        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=class_weight,
        )
        model.fit(X_train_scaled, y_train)
        y_validation_pred = model.predict(X_validation_scaled)

        metrics = classification_metrics_summary(
            y_validation,
            y_validation_pred,
            CLASS_NAMES,
        )
        fold_metrics.append(metrics)
        fold_scores.append(_score_from_metrics(metrics, score_mode))

    return float(np.mean(fold_scores)), _average_metrics(fold_metrics)


def _failure_result(
    f_low,
    f_high,
    window_size,
    C,
    gamma,
    overlap=0.0,
    class_weight="none",
    feature_setting="entropy_hjorth",
    score_mode="accuracy",
    use_cv=False,
    cv_folds=5,
):
    """Return a details dictionary for invalid or failed parameter sets."""
    return {
        "score": 0.0,
        "validation_metrics": _empty_metrics(),
        "test_metrics": _empty_metrics(),
        "parameters": {
            "f_low": f_low,
            "f_high": f_high,
            "window_size": window_size,
            "overlap": overlap,
            "C": C,
            "gamma": gamma,
            "class_weight": class_weight,
            "feature_setting": feature_setting,
            "score_mode": score_mode,
            "use_cv": use_cv,
            "cv_folds": cv_folds,
        },
        "feature_matrix_shape": None,
        "class_distribution": {},
    }


def _validate_parameters(fs, f_low, f_high, window_size, overlap, C, gamma, class_weight):
    """Check that one candidate parameter set is valid."""
    if f_low >= f_high:
        raise ValueError("f_low must be smaller than f_high.")
    if f_high >= fs / 2.0:
        raise ValueError("f_high must be smaller than the Nyquist frequency fs / 2.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in the range [0, 1).")
    if C <= 0:
        raise ValueError("C must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    _normalise_class_weight(class_weight)


def evaluate_eeg_svm_params(
    raw_signals,
    raw_labels,
    fs,
    f_low,
    f_high,
    window_size,
    C,
    gamma,
    overlap=0.0,
    class_weight="none",
    feature_setting="entropy_hjorth",
    score_mode="accuracy",
    use_cv=False,
    cv_folds=5,
    return_details=False,
):
    """
    Evaluate one EEG-SVM parameter combination.

    Objective function:
        parameters
        -> band-pass filtering
        -> window segmentation
        -> feature extraction
        -> train/validation/test split or stratified K-fold CV
        -> StandardScaler fitted on each training set only
        -> train RBF-SVM on each training set
        -> evaluate on validation set/folds
        -> return the selected validation objective score

    Bayesian Optimization and Random Search use the selected validation score
    to choose parameters. The test set is kept for final unbiased evaluation.
    """
    try:
        _validate_parameters(fs, f_low, f_high, window_size, overlap, C, gamma, class_weight)
        sklearn_class_weight = _normalise_class_weight(class_weight)

        filtered_signals = bandpass_filter_signals(raw_signals, fs, f_low, f_high)
        windows, window_labels = segment_windows(
            filtered_signals,
            raw_labels,
            fs,
            window_size,
            overlap=overlap,
        )
        features = extract_features_by_setting(windows, fs, feature_setting)

        if use_cv:
            score, validation_metrics = _cross_validate_svm(
                features,
                window_labels,
                C=C,
                gamma=gamma,
                class_weight=sklearn_class_weight,
                score_mode=score_mode,
                cv_folds=cv_folds,
            )
        else:
            validation_metrics = None
            score = None

        model_result = train_validate_test_svm(
            features,
            window_labels,
            C=C,
            gamma=gamma,
            class_names=CLASS_NAMES,
            class_weight=sklearn_class_weight,
        )

        if not use_cv:
            validation_metrics = model_result["validation_metrics"]
            score = _score_from_metrics(validation_metrics, score_mode)

        details = {
            "score": float(score),
            "validation_metrics": validation_metrics,
            "test_metrics": model_result["test_metrics"],
            "parameters": {
                "f_low": float(f_low),
                "f_high": float(f_high),
                "window_size": float(window_size),
                "overlap": float(overlap),
                "C": float(C),
                "gamma": float(gamma),
                "class_weight": class_weight,
                "feature_setting": feature_setting,
                "score_mode": score_mode,
                "use_cv": use_cv,
                "cv_folds": cv_folds,
            },
            "feature_matrix_shape": features.shape,
            "class_distribution": {
                int(label): int(count)
                for label, count in zip(*np.unique(window_labels, return_counts=True))
            },
        }

        if return_details:
            return details
        return float(score)

    except Exception as exc:
        print(
            "Warning: EEG-SVM evaluation failed for "
            f"f_low={f_low}, f_high={f_high}, window_size={window_size}, "
            f"overlap={overlap}, C={C}, gamma={gamma}, class_weight={class_weight}, "
            f"feature_setting={feature_setting}. Reason: {exc}"
        )
        details = _failure_result(
            f_low,
            f_high,
            window_size,
            C,
            gamma,
            overlap,
            class_weight,
            feature_setting,
            score_mode,
            use_cv,
            cv_folds,
        )
        if return_details:
            return details
        return 0.0
