"""
data_loader.py

Load the Bonn EEG dataset for the five-class problem.

Expected labels:
    Z = 0, O = 1, N = 2, F = 3, S = 4
"""

from pathlib import Path

import numpy as np


CLASS_LABELS = {
    "Z": 0,
    "O": 1,
    "N": 2,
    "F": 3,
    "S": 4,
}

SET_NAME_LABELS = {
    "A": 0,
    "SETA": 0,
    "SET A": 0,
    "B": 1,
    "SETB": 1,
    "SET B": 1,
    "C": 2,
    "SETC": 2,
    "SET C": 2,
    "D": 3,
    "SETD": 3,
    "SET D": 3,
    "E": 4,
    "SETE": 4,
    "SET E": 4,
}


def _normalise_name(name):
    """Normalise folder names such as 'set A', 'Set_A', and 'set-A'."""
    return name.upper().replace("_", " ").replace("-", " ").strip()


def _infer_class_from_path(file_path, dataset_path):
    """Infer the Bonn class from the file name or one of its parent folders."""
    stem = file_path.stem
    file_prefix = stem[:1].upper()
    next_char = stem[1:2]
    if file_prefix in CLASS_LABELS and (not next_char or next_char.isdigit() or next_char in "_- "):
        return CLASS_LABELS[file_prefix]

    relative_parts = file_path.relative_to(dataset_path).parts[:-1]
    for part in reversed(relative_parts):
        normalised = _normalise_name(part)
        compact = normalised.replace(" ", "")

        if normalised in CLASS_LABELS:
            return CLASS_LABELS[normalised]
        if compact in SET_NAME_LABELS:
            return SET_NAME_LABELS[compact]
        if normalised in SET_NAME_LABELS:
            return SET_NAME_LABELS[normalised]

    return None


def _load_signal_file(file_path):
    """Load one Bonn text file as a one-dimensional signal."""
    try:
        signal = np.loadtxt(file_path, dtype=float)
    except ValueError:
        signal = np.loadtxt(file_path, dtype=float, delimiter=",")
    return np.asarray(signal, dtype=float).reshape(-1)


def load_bonn_dataset(dataset_path, classification="five-class"):
    """
    Load Bonn EEG text files from a dataset folder.

    The loader accepts common layouts such as:
        Dataset/Z/Z001.txt
        Dataset/O/O001.txt
        Dataset/Z001.txt

    Parameters
    ----------
    dataset_path : str or Path
        Folder containing the Bonn EEG text files.
    classification : str
        Only "five-class" is supported in this project stage.
    """
    if classification != "five-class":
        raise ValueError("Only classification='five-class' is supported.")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    text_files = sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".asc", ".csv"}
    )

    signals = []
    labels = []

    for file_path in text_files:
        label = _infer_class_from_path(file_path, dataset_path)
        if label is None:
            continue

        try:
            signals.append(_load_signal_file(file_path))
            labels.append(label)
        except Exception as exc:
            print(f"Warning: could not read {file_path}: {exc}")

    if not signals:
        raise ValueError(
            "No Bonn EEG files were loaded. Use file or folder names starting "
            "with Z, O, N, F, or S."
        )

    return signals, np.asarray(labels, dtype=int)
