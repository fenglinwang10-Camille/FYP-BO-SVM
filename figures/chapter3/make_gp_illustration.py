"""Original one-dimensional GP illustration; synthetic values, not FYP results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
x = np.linspace(0, 1, 700)
observed_x = np.array([0.12, 0.36, 0.64, 0.83])
observed_y = np.array([0.65, -0.70, -0.20, 0.55])
length_scale, amplitude, noise_sd = 0.18, 0.65, 0.12

def kernel(a, b):
    r = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]) / length_scale
    return amplitude**2 * (1 + np.sqrt(5)*r + 5*r*r/3) * np.exp(-np.sqrt(5)*r)

K = kernel(observed_x, observed_x) + noise_sd**2 * np.eye(len(observed_x))
L = np.linalg.cholesky(K)
cross = kernel(observed_x, x)
mean = cross.T @ np.linalg.solve(L.T, np.linalg.solve(L, observed_y))
v = np.linalg.solve(L, cross)
variance = amplitude**2 - np.sum(v*v, axis=0)
assert variance.min() > 0
assert variance.max() <= amplitude**2 + 1e-12
sd = np.sqrt(variance)

plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':10,
                     'mathtext.fontset':'dejavuserif', 'pdf.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 1, figsize=(7, 5.9), sharex=True, sharey=True)
blue = '#0072B2'
for ax, centre, spread, title in zip(axes, [np.zeros_like(x), mean],
                                    [np.full_like(x, amplitude), sd],
                                    ['(a) Prior', '(b) Posterior after four observations']):
    ax.fill_between(x, centre-1.96*spread, centre+1.96*spread, color=blue, alpha=0.17, linewidth=0)
    ax.plot(x, centre, color=blue, linewidth=1.8)
    ax.set_title(title, loc='left', fontsize=11, pad=9)
    ax.set_ylabel(r'Latent response $h(u)$')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.set_axisbelow(True)
axes[1].scatter(observed_x, observed_y, s=27, color='black', zorder=4)
axes[1].set_xlabel(r'Search coordinate $u$')
legend = [Line2D([0],[0],color=blue,lw=1.8,label='Mean'),
          Patch(facecolor=blue,alpha=0.17,label='95% pointwise credible interval'),
          Line2D([0],[0],marker='o',color='black',lw=0,markersize=4,label='Observations')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(0.5,0.995),ncol=3,
           frameon=False,fontsize=9,handlelength=1.8,columnspacing=1.3)
fig.subplots_adjust(top=0.88,bottom=0.16,left=0.12,right=0.98,hspace=0.37)
fig.text(0.5,0.035,'One-dimensional illustration with synthetic observations; not experimental results.',
         ha='center',fontsize=8,color='0.35')
for extension in ['pdf','svg','png']:
    fig.savefig(OUT / f'gp_prior_posterior.{extension}',dpi=400,facecolor='white')
plt.close(fig)

caption = r'''\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/chapter3/gp_prior_posterior.pdf}
    \caption{A one-dimensional illustration of GP conditioning with a Mat\'ern $5/2$ kernel. Panel (a) shows the prior mean and 95\% pointwise credible intervals. Panel (b) shows the posterior after conditioning on four synthetic observations under an independent Gaussian residual model. The shaded posterior interval represents uncertainty about the latent response, rather than a future noisy observation. The covariance parameters are held fixed between panels. This illustration is not an experimental result.}
    \label{fig:gp-prior-posterior}
\end{figure}
'''
(OUT/'gp_prior_posterior.tex').write_text(caption,encoding='utf-8')
(OUT/'README.txt').write_text('Original illustration inspired by the prior/posterior presentation in Garnett, Bayesian Optimization (2023), Figures 2.1 and 2.2. Not reproduced from the book.\nMatérn nu=2.5; length scale=0.18; prior standard deviation=0.65; residual standard deviation=0.12; zero prior mean. All values are illustrative. Intervals describe latent pointwise uncertainty.\n',encoding='utf-8')
print(OUT / 'gp_prior_posterior.png')
