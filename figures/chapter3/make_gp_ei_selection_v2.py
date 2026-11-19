"""Original GP/EI illustration using synthetic observations, not EEG results."""
from pathlib import Path
from math import erf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
u = np.linspace(0, 1, 2001)
observed_u = np.array([0.08, 0.32, 0.61, 0.90])
observed_y = np.array([-0.32, -0.72, -0.48, -0.30])
prior_mean = -0.45
amplitude, length_scale, noise_sd, xi = 0.24, 0.18, 0.015, 0.01

def kernel(a, b):
    r = np.abs(a[:, None] - b[None, :]) / length_scale
    return amplitude**2 * (1 + np.sqrt(5)*r + 5*r*r/3) * np.exp(-np.sqrt(5)*r)

covariance = kernel(observed_u, observed_u) + noise_sd**2 * np.eye(len(observed_u))
chol = np.linalg.cholesky(covariance)
cross = kernel(observed_u, u)
weights = np.linalg.solve(chol.T, np.linalg.solve(chol, observed_y-prior_mean))
mean = prior_mean + cross.T @ weights
projected = np.linalg.solve(chol, cross)
variance = amplitude**2 - np.sum(projected**2, axis=0)
assert variance.min() >= -1e-12
sd = np.sqrt(np.maximum(variance, 0))
delta = observed_y.min() - xi - mean
z = np.divide(delta, sd, out=np.zeros_like(delta), where=sd>0)
cdf = 0.5 * (1 + np.array([erf(value/np.sqrt(2)) for value in z]))
pdf = np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
ei = np.where(sd>0, delta*cdf + sd*pdf, np.maximum(delta, 0))
assert np.all(np.isfinite(ei)) and ei.min() >= -1e-12
selected_index = int(np.argmax(ei))
next_u = u[selected_index]

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7,
})

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
blue, orange = "#365F82", "#B86B32"
fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.9))
left.fill_between(u, mean-1.96*sd, mean+1.96*sd,
                  color=blue, alpha=0.20, linewidth=0)
left.plot(u, mean, color=blue, lw=1.8)
left.scatter(observed_u, observed_y, s=30, color="#222222", zorder=5)
left.set_ylabel(r"Objective value (lower is better)")
left.set_title("(a) GP prediction", loc="left", fontsize=10, pad=9)
left.set_ylim(-1.07, 0.02)
minimum_index = int(np.argmin(mean))
left.scatter([u[minimum_index]], [mean[minimum_index]], marker="D", s=28,
             facecolors="white", edgecolors=blue, zorder=6)
left.annotate("Lowest predicted\nmean", xy=(u[minimum_index], mean[minimum_index]),
              xytext=(0.02, -0.97), fontsize=8, color=blue,
              arrowprops={"arrowstyle":"->", "color":blue, "lw":0.9})
left.scatter([next_u], [mean[selected_index]], s=48, facecolors="white",
             edgecolors=orange, linewidths=1.6, zorder=7)
left.annotate("Selected location\n(not yet evaluated)",
              xy=(next_u, mean[selected_index]), xytext=(0.55, -0.92),
              fontsize=7.8, color=orange,
              arrowprops={"arrowstyle":"->", "color":orange, "lw":0.9})
right.plot(u, ei, color=orange, lw=1.9)
right.scatter([next_u], [ei[selected_index]], s=45, color=orange, zorder=5)
right.set_ylabel(r"Expected improvement $\mathrm{EI}_t(u)$")
right.set_title("(b) Select the maximum of EI", loc="left", fontsize=10, pad=9)
right.set_ylim(0, ei.max()*1.28)
right.annotate("Maximum EI\nNext evaluation", xy=(next_u, ei[selected_index]),
               xytext=(0.59, 0.025), fontsize=8.5, color=orange,
               arrowprops={"arrowstyle":"->", "color":orange, "lw":1})
for ax in (left, right):
    ax.axvline(next_u, color=orange, ls="--", lw=1.1, zorder=2)
    ax.set_xlabel(r"Search coordinate $u$")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#E5E8EB", lw=0.55)
    ax.set_axisbelow(True)
    ax.text(next_u+0.018, 0.965, r"$u_{t+1}$", color=orange,
            transform=ax.get_xaxis_transform(), fontsize=10, va="top")
legend = [Line2D([0],[0], color=blue, lw=1.8, label=r"GP mean $\mu_t(u)$"),
          Patch(facecolor=blue, alpha=0.20, label=r"95% uncertainty band: $\mu_t(u)\pm1.96s_t(u)$"),
          Line2D([0],[0], marker="o", color="#222222", lw=0, markersize=4,
                 label=r"Observed values $(u_i,y_i)$")]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.52, 1.0),
           ncol=3, frameon=False, fontsize=7.7, columnspacing=1.3, handlelength=1.7)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.80, wspace=0.39)
fig.text(0.52, 0.025,
         r"Both dashed lines mark the same selected coordinate $u_{t+1}$.",
         ha="center", fontsize=8.5, color=orange)
for extension in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"gp_ei_selection_u_v2.{extension}", dpi=400, facecolor="white")
plt.close(fig)
print(f"Synthetic illustration: mean minimum u={u[minimum_index]:.4f}; next u={next_u:.4f}")
