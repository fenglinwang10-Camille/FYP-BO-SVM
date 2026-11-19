"""
plot_unfiltered_average_psd.py

Inspect the unfiltered Bonn EEG signals by plotting the class-wise average PSD
and checking for a narrow 50 Hz power-line peak.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from data_loader import load_bonn_dataset
from display_utils import display_dataframe, display_image


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Dataset"
OUTPUT_DIR = PROJECT_DIR / "results" / "raw_psd_inspection"
FS = 173.61
CLASS_ORDER = ["Z", "O", "N", "F", "S"]
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_ORDER)}


def _compute_average_psd(signals):
    """Return a common frequency grid and the mean Welch PSD over signals."""
    psd_rows = []
    frequency_grid = None

    for signal in signals:
        freqs, psd = welch(
            np.asarray(signal, dtype=float),
            fs=FS,
            nperseg=min(1024, len(signal)),
            noverlap=512,
            scaling="density",
        )
        if frequency_grid is None:
            frequency_grid = freqs
        psd_rows.append(psd)

    return frequency_grid, np.mean(np.asarray(psd_rows), axis=0)


def _summarise_50hz_peak(freqs, average_psd):
    """
    Quantify the prominence of the 50 Hz neighborhood against nearby sidebands.
    """
    peak_mask = (freqs >= 48.0) & (freqs <= 52.0)
    left_mask = (freqs >= 45.0) & (freqs < 48.0)
    right_mask = (freqs > 52.0) & (freqs <= 55.0)
    baseline_mask = left_mask | right_mask

    peak_freqs = freqs[peak_mask]
    peak_values = average_psd[peak_mask]
    baseline_values = average_psd[baseline_mask]

    peak_index = int(np.argmax(peak_values))
    peak_frequency = float(peak_freqs[peak_index])
    peak_power = float(peak_values[peak_index])
    baseline_mean = float(np.mean(baseline_values))
    baseline_median = float(np.median(baseline_values))
    prominence_ratio = float(peak_power / (baseline_mean + 1e-12))

    return {
        "peak_freq_48_52_hz": peak_frequency,
        "peak_power_48_52_hz": peak_power,
        "sideband_mean_45_48_52_55_hz": baseline_mean,
        "sideband_median_45_48_52_55_hz": baseline_median,
        "peak_to_sideband_ratio": prominence_ratio,
    }


def _plot_average_psd(class_psd):
    """Save a full-spectrum class-wise average PSD plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for class_name in CLASS_ORDER:
        freqs, average_psd = class_psd[class_name]
        ax.plot(freqs, average_psd, linewidth=2, label=class_name)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average PSD")
    ax.set_title("Unfiltered Bonn EEG: Class-wise Average PSD")
    ax.set_xlim(0, 80)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Class")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "average_psd_unfiltered.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _plot_50hz_zoom(class_psd):
    """Save a zoomed PSD plot around the 50 Hz neighborhood."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for class_name in CLASS_ORDER:
        freqs, average_psd = class_psd[class_name]
        zoom_mask = (freqs >= 45.0) & (freqs <= 55.0)
        ax.plot(freqs[zoom_mask], average_psd[zoom_mask], linewidth=2, label=class_name)

    ax.axvspan(48.0, 52.0, color="#f59e0b", alpha=0.12, label="48-52 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average PSD")
    ax.set_title("Unfiltered Bonn EEG: 50 Hz Neighborhood")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Class")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "average_psd_unfiltered_45_55hz.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signals, labels = load_bonn_dataset(DATASET_DIR, classification="five-class")

    class_psd = {}
    peak_rows = []

    for class_name in CLASS_ORDER:
        class_label = CLASS_TO_LABEL[class_name]
        class_signals = [signal for signal, label in zip(signals, labels) if label == class_label]
        freqs, average_psd = _compute_average_psd(class_signals)
        class_psd[class_name] = (freqs, average_psd)

        summary = _summarise_50hz_peak(freqs, average_psd)
        summary["class"] = class_name
        peak_rows.append(summary)

    summary_df = pd.DataFrame(peak_rows)[
        [
            "class",
            "peak_freq_48_52_hz",
            "peak_power_48_52_hz",
            "sideband_mean_45_48_52_55_hz",
            "sideband_median_45_48_52_55_hz",
            "peak_to_sideband_ratio",
        ]
    ]

    full_plot_path = _plot_average_psd(class_psd)
    zoom_plot_path = _plot_50hz_zoom(class_psd)
    summary_path = OUTPUT_DIR / "average_psd_50hz_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    display_dataframe(summary_df)
    display_image(full_plot_path)
    display_image(zoom_plot_path)

    print(summary_df.to_string(index=False))
    print(f"Saved {full_plot_path}")
    print(f"Saved {zoom_plot_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
