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
blue, orange = "#365F82", "#B86B32"
fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.5))
left.fill_between(u, mean-1.96*sd, mean+1.96*sd,
                  color=blue, alpha=0.16, linewidth=0,
                  label="95% pointwise credible interval")
left.plot(u, mean, color=blue, lw=1.6, label=r"Posterior mean $\mu_t(u)$")
left.scatter(observed_u, observed_y, s=22, color="#222222", zorder=4,
             label=r"Observations $(u_i,y_i)$")
left.set_ylabel(r"Latent objective $h(u)$")
left.set_title("(a) GP prediction", loc="left", fontsize=10)
left.set_ylim(-1.02, 0.12)
left.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.8)

right.plot(u, ei, color=blue, lw=1.6, label=r"$\mathrm{EI}_t(u)$")
right.scatter([next_u], [ei[selected_index]], s=32, color=orange, zorder=4,
              label=r"Next evaluation $u_{t+1}$")
right.set_ylabel(r"Expected improvement $\mathrm{EI}_t(u)$")
right.set_title("(b) EI-based selection", loc="left", fontsize=10)
right.set_ylim(0, ei.max()*1.36)
right.ticklabel_format(axis="y", style="plain", useOffset=False)
right.legend(loc="upper right", frameon=False, fontsize=8)
for ax in (left, right):
    ax.axvline(next_u, ymax=0.72 if ax is left else 0.80,
               color=orange, ls="--", lw=1.0, zorder=2)
    ax.set_xlabel(r"Search coordinate $u$")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color="#E5E8EB", lw=0.55)
    ax.set_axisbelow(True)
    ax.text(next_u+0.012, 0.025, r"$u_{t+1}$", color=orange,
            transform=ax.get_xaxis_transform(), fontsize=9)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.88, wspace=0.40)
for extension in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"gp_ei_selection_u_clean.{extension}", dpi=400, facecolor="white")
plt.close(fig)
print(f"Synthetic illustration: next u={next_u:.4f}; max EI={ei[selected_index]:.6f}")
print(OUT / "gp_ei_selection_u_clean.png")
