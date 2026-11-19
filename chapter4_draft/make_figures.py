"""Export exact, editable methods diagrams; no experiment is rerun."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures" / "chapter4"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
NAVY, BLUE, TEAL, ORANGE = "#16324F", "#E9F1FA", "#DDF1EA", "#FCE8CF"
INK, GRAY = "#203445", "#667788"

def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax

def box(ax, x, y, w, h, text, color=BLUE, size=11, edge=NAVY, weight="normal"):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle="round,pad=0.025,rounding_size=0.10",
                 facecolor=color, edgecolor=edge, linewidth=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=INK, linespacing=1.4, fontweight=weight, zorder=4)

def arrow(ax, points, color=NAVY, lw=1.5):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, zorder=2))

def label(ax, x, y, text, size=10, color=GRAY, **kw):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, **kw)

def save(fig, stem):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT/f"{stem}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)

def implementation():
    fig, ax = setup(16, 10)
    label(ax, 8, 9.65, "Implementation of the EEG optimisation pipeline", 19, NAVY,
          fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.35, 4.0), 15.3, 5.0,
                 boxstyle="round,pad=0.02,rounding_size=0.15", facecolor="#F8FAFD",
                 edgecolor="#8CA5C0", linestyle="--", lw=1.3))
    label(ax, 8, 8.78, "CANDIDATE EVALUATION: OUTER TRAINING RECORDINGS ONLY", 12, NAVY,
          fontweight="bold")
    box(ax, 2.1, 7.85, 2.9, 0.9, "BO optimiser\nGP surrogate + EI", weight="bold")
    box(ax, 7.8, 7.85, 3.5, 0.9, "Candidate configuration\nf_low, f_high, duration, C, gamma")
    arrow(ax, [(3.55, 7.85), (6.05, 7.85)])
    label(ax, 12.7, 7.95, "50 candidate evaluations\n16 initial + 34 guided", 11)
    xs = [1.9, 4.95, 8, 11.05, 14.1]
    texts = ["Band-pass filter\n+ non-overlapping\nwindows", "Fixed 55-feature\nextraction", "Fold-local scaler\n+ RBF-SVM", "Window predictions\n+ majority vote\nper recording", "Mean recording-level\nCV accuracy"]
    for x, t in zip(xs, texts):
        box(ax, x, 5.35, 2.65, 1.15, t, ORANGE if x == xs[-1] else BLUE, size=10.8)
    for x, q in zip(xs, xs[1:]):
        arrow(ax, [(x+1.325, 5.35), (q-1.325, 5.35)])
    arrow(ax, [(6.7, 7.4), (6.7, 6.9), (1.9, 6.9), (1.9, 5.925)])
    label(ax, 3.95, 7.10, "f_low, f_high, duration", 10)
    arrow(ax, [(8, 7.4), (8, 5.925)])
    label(ax, 8.55, 6.85, "C, gamma", 10)
    label(ax, 4.95, 6.12, "Training EEG input", 10)
    arrow(ax, [(4.1, 6.12), (0.48, 6.12), (0.48, 5.35), (0.575, 5.35)])
    label(ax, 9.55, 4.52, "Fixed five recording folds; refit scaler and SVM in every fold", 10)
    arrow(ax, [(14.1, 5.925), (14.1, 7.05), (15.35, 7.05),
               (15.35, 8.46), (2.1, 8.46), (2.1, 8.30)], color="#B46A14")
    label(ax, 11.65, 8.58, "Validation feedback", 10, "#965A12")
    label(ax, 8, 3.53, "AFTER SEARCH: SELECT BY CV, REFIT, THEN TEST", 12, "#16624D",
          fontweight="bold")
    box(ax, 2, 2.52, 2.8, 1.05, "Best candidate\nby CV accuracy", TEAL)
    box(ax, 6, 2.52, 3.25, 1.05, "Refit scaler + SVM\non windows from all\n400 training recordings", TEAL, 10.8)
    box(ax, 10.15, 2.52, 3.15, 1.05, "Predict test windows\n+ majority vote\nper recording", TEAL, 10.8)
    box(ax, 14, 2.52, 2.9, 1.05, "Recording-level\ntest metrics", ORANGE)
    for a,b in [(3.4,4.375),(7.625,8.575),(11.725,12.55)]:
        arrow(ax, [(a,2.52),(b,2.52)], color="#16624D")
    box(ax, 10.15, 0.85, 4.25, 0.82,
        "100 held-out recordings\nSelected preprocessing + 55 features", TEAL, 10.5)
    arrow(ax, [(10.15,1.26),(10.15,1.995)], color="#16624D")
    label(ax, 4.4, 0.95, "No test feedback to BO.\nOne held-out evaluation per selected configuration.", 10)
    save(fig, "fig4_1_implementation")

def partition():
    fig, ax = setup(15, 11.3)
    label(ax, 7.5, 10.95, "Recording-level data partition and validation", 19, NAVY,
          fontweight="bold")
    box(ax, 7.5, 10.0, 7.2, 0.95,
        "Bonn EEG: 500 original recordings\n100 per class: Z, O, N, F, S", size=12, weight="bold")
    box(ax, 5.2, 8.25, 6.3, 0.9,
        "Outer training set: 400 recordings\n80 per class", size=12)
    box(ax, 12.1, 8.25, 4.0, 0.9,
        "Held-out test: 100 recordings\n20 per class", TEAL, 11.5)
    arrow(ax, [(7.5,9.525),(7.5,9.1),(5.2,9.1),(5.2,8.7)])
    arrow(ax, [(7.5,9.1),(12.1,9.1),(12.1,8.7)], color="#16624D")
    label(ax, 5.2, 9.24, "Stratified 80/20 split BEFORE windowing", 10)
    label(ax, 5.35, 7.37, "FIXED RECORDING-LEVEL FIVE-FOLD CV", 12, NAVY,
          fontweight="bold")
    x0, y0, cw, ch = 2.1, 6.37, 1.25, 0.45
    for j in range(5):
        label(ax,x0+(j+.5)*cw,6.9,f"Group {j+1}\n80 records",9)
    for i in range(5):
        y = y0-i*ch
        label(ax,1.55,y+ch/2,f"CV {i+1}",10)
        for j in range(5):
            ax.add_patch(Rectangle((x0+j*cw,y),cw-.035,ch-.035,
                         facecolor=ORANGE if i==j else BLUE,edgecolor="white"))
            label(ax,x0+(j+.5)*cw-.017,y+ch/2-.017,
                  "Validation" if i==j else "Train",9,INK)
        label(ax,9.4,y+ch/2,"320 train / 80 validation",9)
    label(ax,5.4,4.28,"Each recording's windows remain together in one fold",10,NAVY)
    box(ax,3.15,3.28,3.55,.95,"Fit scaler + RBF-SVM\non fold training windows",size=10.5)
    box(ax,7.4,3.28,3.55,.95,"Predict validation windows\n+ recording majority vote",size=10.5)
    arrow(ax,[(4.925,3.28),(5.625,3.28)])
    box(ax,5.2,1.94,6.4,.85,"Mean recording-level CV accuracy\nCandidate selection uses this score only",ORANGE,11)
    arrow(ax,[(7.4,2.805),(7.4,2.365)])
    arrow(ax,[(3.15,4.02),(3.15,3.755)])
    arrow(ax,[(12.1,7.8),(12.1,1.20)],color="#16624D")
    label(ax,12.1,6.8,"Isolated during search",10,"#16624D")
    box(ax,5.2,.73,6.5,.85,"Select by CV; refit on all 400 training records",TEAL,10.8)
    arrow(ax,[(5.2,1.515),(5.2,1.155)],color="#16624D")
    box(ax,12.1,.73,4.65,.85,"Final test: 100 recordings\nOne evaluation of selected configuration",TEAL,10)
    arrow(ax,[(8.45,.73),(9.775,.73)],color="#16624D")
    label(ax,7.5,.12,"Recording-level separation does not establish subject-independent evaluation.",10)
    save(fig,"fig4_2_recording_split")

if __name__ == "__main__":
    implementation()
    partition()
    print(f"Exported two figures as PNG, SVG and PDF to {OUT}")
