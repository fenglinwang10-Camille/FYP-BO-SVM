"""
metrics.py

Evaluation helpers for the five-class Bonn EEG project.

Accuracy is one overall score. Per-class recall reports the detection rate for
each class separately. Seizure sensitivity is highlighted only for class S,
because S represents seizure EEG in the Bonn dataset.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


def compute_accuracy(y_true, y_pred):
    """Return the overall correct classification rate."""
    return float(accuracy_score(y_true, y_pred))


def compute_per_class_recall(y_true, y_pred, class_names):
    """
    Return recall rate for each class separately.

    Recall for one class is TP / (TP + FN). This is different from accuracy:
    accuracy is one overall score, while per-class recall shows how well each
    individual EEG class is detected.
    """
    labels = list(range(len(class_names)))
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        class_name: float(recall_value)
        for class_name, recall_value in zip(class_names, recalls)
    }


def compute_macro_recall(y_true, y_pred, class_names):
    """Return the unweighted mean recall across all classes."""
    labels = list(range(len(class_names)))
    return float(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def compute_seizure_sensitivity(y_true, y_pred, seizure_label=4):
    """
    Return seizure sensitivity for class S only.

    In the Bonn dataset, S represents seizure EEG. Therefore seizure
    sensitivity is Recall_S = TP_S / (TP_S + FN_S), not a separate sensitivity
    value for every class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_seizure = y_true == seizure_label
    tp = np.sum(true_seizure & (y_pred == seizure_label))
    fn = np.sum(true_seizure & (y_pred != seizure_label))

    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def compute_confusion_matrix(y_true, y_pred):
    """Return a 5x5 confusion matrix using labels 0 to 4."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])


def classification_metrics_summary(y_true, y_pred, class_names):
    """Return all current project metrics in one dictionary."""
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_recall": compute_macro_recall(y_true, y_pred, class_names),
        "per_class_recall": compute_per_class_recall(y_true, y_pred, class_names),
        "seizure_sensitivity": compute_seizure_sensitivity(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }
