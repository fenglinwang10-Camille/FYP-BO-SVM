import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


TITLE_BLUE = "#3437B7"
MAIN_BLUE = "#2B5DAA"
SOFT_FILL = "#F3F4FA"
ORANGE = "#E08A2E"
TEXT = "#111111"


def add_round_box(ax, xy, width, height, text, fontsize=18, lw=2.2,
                  facecolor=SOFT_FILL, edgecolor=MAIN_BLUE, header=None,
                  header_face=TITLE_BLUE, text_color=TEXT):
    x, y = xy
    body = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor
    )
    ax.add_patch(body)

    if header:
        header_h = 0.18 * height
        hdr = FancyBboxPatch(
            (x, y + height - header_h), width, header_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0, edgecolor=header_face, facecolor=header_face
        )
        ax.add_patch(hdr)
        ax.text(
            x + 0.02 * width, y + height - header_h / 2,
            header, ha="left", va="center",
            fontsize=fontsize, color="white", fontweight="bold"
        )
        text_y = y + 0.43 * height
    else:
        text_y = y + 0.5 * height

    ax.text(
        x + width / 2, text_y, text,
        ha="center", va="center",
        fontsize=fontsize, color=text_color, wrap=True
    )


fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Core idea block
core_text = (
    "Bayesian optimisation combines a Gaussian Process surrogate model with an\n"
    "acquisition function to select promising parameter settings based on the\n"
    "results of previous evaluations."
)
add_round_box(
    ax, (0.05, 0.73), 0.90, 0.18, core_text,
    fontsize=16, header="Core Idea"
)

# Flow steps
step_x = 0.18
step_w = 0.58
step_h = 0.075
step_y = [0.60, 0.49, 0.38, 0.27, 0.16]
step_texts = [
    "Evaluate 16 initial parameter configurations",
    "Fit a Gaussian Process to the observed parameter-score pairs",
    "Use Expected Improvement to select the next configuration",
    "Evaluate the configuration using five-fold grouped cross-validation",
    "Add the new result and update the Gaussian Process",
]

for y, txt in zip(step_y, step_texts):
    add_round_box(
        ax, (step_x, y), step_w, step_h, txt,
        fontsize=16, lw=1.8, facecolor="white", edgecolor=MAIN_BLUE
    )

# Vertical arrows
for i in range(len(step_y) - 1):
    start = (step_x + step_w / 2, step_y[i])
    end = (step_x + step_w / 2, step_y[i + 1] + step_h)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=2.0, color=MAIN_BLUE
    ))

# Loop arrow from last step back to third step
loop_start = (step_x + step_w, step_y[-1] + step_h / 2)
loop_mid1 = (0.86, step_y[-1] + step_h / 2)
loop_mid2 = (0.86, step_y[2] + step_h / 2 + 0.11)
loop_end = (step_x + step_w, step_y[2] + step_h / 2)

ax.plot(
    [loop_start[0], loop_mid1[0], loop_mid2[0], loop_end[0]],
    [loop_start[1], loop_mid1[1], loop_mid2[1], loop_end[1]],
    linestyle=(0, (3, 3)), linewidth=2.0, color=ORANGE
)
ax.add_patch(FancyArrowPatch(
    (0.84, step_y[2] + step_h / 2 + 0.085), loop_end,
    arrowstyle="-|>", mutation_scale=18, linewidth=0, color=ORANGE
))

ax.text(
    0.89, 0.30,
    "Repeat Steps 3-5\nuntil the 50-evaluation\nbudget is exhausted",
    ha="left", va="center", fontsize=16, color=ORANGE, fontweight="bold"
)

out_path = r"C:\Users\王枫淋\Desktop\FYP\results\04_grouped_bo_svm\bo_optimisation_flowchart.png"
plt.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(out_path)
