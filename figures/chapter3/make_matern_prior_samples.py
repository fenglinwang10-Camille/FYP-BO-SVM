"""Illustrative latent GP prior draws, not EEG data or fitted BO results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
u = np.linspace(0.0, 1.0, 400)
length_scales = (0.08, 0.20, 0.50)
variance = 1.0
jitter = 1e-10  # Numerical Cholesky stabilisation, not observation noise.
rng = np.random.default_rng(42)
z = rng.standard_normal((len(u), 3))  # Shared draws permit a controlled comparison.
samples = []
for ell in length_scales:
    r = np.abs(u[:, None] - u[None, :]) / ell
    K = variance * (1 + np.sqrt(5)*r + 5*r*r/3) * np.exp(-np.sqrt(5)*r)
    assert np.allclose(np.diag(K), variance)
    L = np.linalg.cholesky(K + jitter*np.eye(len(u)))
    draws = L @ z
    assert np.isfinite(draws).all()
    samples.append(draws)

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
                     'mathtext.fontset':'dejavuserif','pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
colours = ['#0072B2', '#D55E00', '#009E73']
styles = ['-', '-', '-']
fig, axes = plt.subplots(3, 1, figsize=(7, 4.2), sharex=True, sharey=True)
limit = max(2.2, np.max(np.abs(np.stack(samples))) * 1.08)
for i, (ax, ell, draws) in enumerate(zip(axes, length_scales, samples)):
    ax.axhline(0, color='0.6', linewidth=0.7, zorder=0)
    for j in range(3):
        ax.plot(u, draws[:, j], color=colours[j], linestyle=styles[j], linewidth=1.25)
    ax.text(0.015, 0.90, f'({chr(97+i)}) '+rf'$\ell={ell:.2f}$',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    ax.set_ylabel(r'$h(u)$')
    ax.set_ylim(-limit, limit)
    ax.set_yticks([-2, 0, 2])
    ax.set_xlim(0,1)
    ax.grid(axis='y', color='0.92', linewidth=0.5)
    ax.set_axisbelow(True)
axes[-1].set_xlabel(r'Search coordinate $u$')
handles = [Line2D([0],[0],color=colours[j],linestyle=styles[j],lw=1.3,label=f'Draw {j+1}') for j in range(3)]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5,0.995),ncol=3,frameon=False)
fig.subplots_adjust(left=0.09,right=0.98,top=0.89,bottom=0.21,hspace=0.15)
fig.text(0.5,0.035,r'Matérn $5/2$ prior: $m(u)=0$, $\sigma_f^2=1$. Synthetic function draws.',
         ha='center',fontsize=8,color='0.35')
for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'matern_prior_samples.{ext}',dpi=400,facecolor='white')
plt.close(fig)
np.savez(OUT/'matern_prior_samples_data.npz',u=u,length_scales=length_scales,samples=np.stack(samples),seed=42,jitter=jitter)
print('Created matern_prior_samples.png, .pdf and .svg')
