"""Plot one reproducibly selected raw segment per Bonn class on shared axes."""

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

    limit = max(np.abs(signal).max() for _, _, _, signal in examples)
    limit = np.ceil(limit * 1.05 / 100) * 100
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(5, 1, figsize=(7.15, 5.6), sharex=True,
                                 sharey=True, layout="constrained")
        for ax, (label, source, time, signal) in zip(axes, examples):
            ax.plot(time, signal, color="#17658a", linewidth=0.7)
            ax.set_ylim(-limit, limit)
            ax.set_xlim(0, duration)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
            ax.grid(axis="y", color="#dfe5e9", linewidth=0.6)
            ax.text(0.99, 0.88, f"{label} ({source.stem})", ha="right",
                    va="top", transform=ax.transAxes, fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xticks(np.arange(6))
        fig.supylabel("Amplitude (original file values)", fontsize=10)
        target = root / "figures" / "chapter4" / "bonn_five_class_signal_examples.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, facecolor="white")
        plt.close(fig)
    for label, source, _, signal in examples:
        print(f"{label}: {source.name}, {len(signal)} points, range {signal.min():g} to {signal.max():g}")
    print(target)


if __name__ == "__main__":
    main()
