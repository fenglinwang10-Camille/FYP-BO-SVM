import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "04_grouped_bo_svm"

RESULT_FILES = {
    "Manual SVM": OUT_DIR / "grouped_manual_final_result.json",
    "Random Search": OUT_DIR / "grouped_random_search_final_result.json",
    "BO-SVM": OUT_DIR / "grouped_bo_final_result.json",
}

classes = ["Z", "O", "N", "F", "S"]
colors = {
    "Manual SVM": "#4C78A8",
    "Random Search": "#F58518",
    "BO-SVM": "#54A24B",
}

recalls = {}
for method, path in RESULT_FILES.items():
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_class = data["window_level_test_metrics"]["per_class_recall"]
    recalls[method] = [100 * per_class[c] for c in classes]

x = np.arange(len(classes))
width = 0.24

fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=240)

offsets = [-width, 0, width]
for offset, (method, values) in zip(offsets, recalls.items()):
    bars = ax.bar(
        x + offset,
        values,
        width,
        label=method,
        color=colors[method],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(
        bars,
        labels=[f"{v:.1f}" for v in values],
        padding=3,
        fontsize=13,
        color="#222222",
    )

ax.set_ylabel("Recall (%)", fontsize=14)
ax.set_xlabel("EEG class", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=14)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylim(0, 112)
ax.grid(axis="y", linestyle="-", alpha=0.28)
ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
    fontsize=12,
    columnspacing=1.6,
    handlelength=1.8,
)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout(rect=[0, 0.06, 1, 1])

out = OUT_DIR / "grouped_classwise_recall_comparison.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")

large_out = OUT_DIR / "grouped_classwise_recall_comparison_large.png"
fig.savefig(large_out, bbox_inches="tight", facecolor="white")

print(out)
print(large_out)
