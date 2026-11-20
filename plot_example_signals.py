"""Plot five raw Bonn examples with a separate amplitude scale for each row."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def main():
    root = Path(__file__).resolve().parent
    fs = 173.61
    duration = 5.0
    examples = []
    for label in ("Z", "O", "N", "F", "S"):
        # A fixed filename rule avoids choosing examples to support a claim.
        source = sorted((root / "Dataset" / label).glob("*.txt"))[0]
        signal = np.loadtxt(source).reshape(-1)
        time = np.arange(len(signal)) / fs
        keep = time < duration
        examples.append((label, source, time[keep], signal[keep]))

    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(5, 1, figsize=(7.15, 3.5), sharex=True)
        fig.subplots_adjust(left=0.13, right=0.91, bottom=0.13, top=0.98,
                            hspace=0.23)
        for ax, (label, source, time, signal) in zip(axes, examples):
            ax.plot(time, signal, color="#0B6798", linewidth=0.65)
            local_limit = max(1, np.abs(signal).max() * 1.10)
            ax.set_ylim(-local_limit, local_limit)
            ax.set_xlim(0, duration)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
            ax.tick_params(labelsize=7.5, length=3)
            ax.grid(axis="y", color="#dfe5e9", linewidth=0.6)
            ax.text(1.015, 0.5, source.stem, ha="left", va="center",
                    fontsize=8.5, transform=ax.transAxes)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xticks(np.arange(6))
        fig.text(0.025, 0.55, "Amplitude (original file values)", rotation=90,
                 va="center", fontsize=9)
        target = root / "figures" / "chapter4" / "bonn_five_class_signal_examples.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, facecolor="white")
        fig.savefig(target.with_name("example_signals.png"), dpi=320, facecolor="white")
        plt.close(fig)
    for label, source, _, signal in examples:
        print(f"{label}: {source.name}, {len(signal)} points, sample SD {np.std(signal, ddof=1):.3f}")
    print(target)


if __name__ == "__main__":
    main()
