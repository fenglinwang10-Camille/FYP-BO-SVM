"""Draw chapter-five PNG figures from completed, saved predictions and histories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42'
OUT = ROOT / 'figures/chapter5'
BLUE, ORANGE, GREEN = '#096398', '#AD4E24', '#42835A'
PARAMS = ['f_low', 'f_high', 'window_size', 'C', 'gamma']


def save(fig, name):
    fig.savefig(OUT/name, dpi=320, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def search_axis(ax):
    ax.axvline(16, color='#67707A', lw=1, ls=(0, (3, 3)))
    ax.set(xlim=(1, 50), xticks=[1, 10, 16, 20, 30, 40, 50],
           xlabel='Candidate evaluations', ylabel='Best inner-CV accuracy (%)')
    ax.grid(axis='y', color='#DFE6EC', lw=.65)


def main():
    OUT.mkdir(exist_ok=True)
    histories = {method: [] for method in ['bo', 'rs']}
    for k in range(1, 6):
        bo = pd.read_csv(RUN/f'outer_fold_{k}/bo_history.csv', float_precision='round_trip')
        rs = pd.read_csv(RUN/f'outer_fold_{k}/rs_history.csv', float_precision='round_trip')
        assert len(bo) == len(rs) == 50
        np.testing.assert_array_equal(bo.loc[:15, PARAMS+['score']], rs.loc[:15, PARAMS+['score']])
        for method, frame in [('bo', bo), ('rs', rs)]:
            histories[method].append(frame.score.to_numpy()*100)
    histories = {k: np.stack(v) for k, v in histories.items()}
    best = {k: np.maximum.accumulate(v, axis=1) for k, v in histories.items()}
    x = np.arange(1, 51)
    summary = json.loads((RUN/'summary.json').read_text())
    results = pd.read_csv(RUN/'outer_fold_results.csv')
    for method in ['bo', 'rs', 'fixed_reference']:
        assert len(results[results.method == method]) == 5

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.axisbelow': True})
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 6.5), layout='constrained')
    colors = ['#096398', '#B4522A', '#12836F', '#835691', '#686C32']
    for k, (values, color) in enumerate(zip(best['bo'], colors), 1):
        axes[0].step(x, values, where='post', lw=1.6, color=color, label=f'Fold {k}')
    search_axis(axes[0])
    axes[0].set_ylim(np.floor(best['bo'].min()/5)*5-1, 90)
    axes[0].set_title('(a) BO search history in each outer fold', loc='left', fontsize=10)
    axes[0].legend(ncols=3, frameon=False, fontsize=8, loc='lower right')
    rng = np.random.default_rng(42)
    positions = np.arange(1, 6)
    for values, offset, color, fill in [(histories['bo'][:, :16], -.18, ORANGE, '#F4E2D8'),
                                       (histories['bo'][:, 16:], .18, BLUE, '#DDECF6')]:
        axes[1].boxplot(values.T, positions=positions+offset, widths=.28,
                        manage_ticks=False, patch_artist=True, showfliers=False,
                        boxprops={'facecolor': fill, 'edgecolor': color},
                        medianprops={'color': color, 'linewidth': 1.5},
                        whiskerprops={'color': color}, capprops={'color': color})
        for position, run in zip(positions+offset, values):
            axes[1].scatter(position+rng.uniform(-.075, .075, len(run)), run,
                            s=10, color=color, alpha=.45, linewidths=0)
    axes[1].set(xlim=(.5, 5.5), ylim=(0, 100), yticks=range(0, 101, 20),
                xticks=positions, xlabel='Outer fold', ylabel='Candidate inner-CV accuracy (%)')
    axes[1].grid(axis='y', color='#DFE6EC', lw=.65)
    axes[1].set_title('(b) BO candidate scores before and after initialisation', loc='left', fontsize=10)
    axes[1].legend(handles=[Patch(facecolor='#F4E2D8', edgecolor=ORANGE, label='Evaluations 1–16'),
                            Patch(facecolor='#DDECF6', edgecolor=BLUE, label='Evaluations 17–50')],
                   loc='lower right', ncols=2, frameon=False, fontsize=8)
    save(fig, 'fig5_1_nested_bo_history_and_phases.png')

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 6.3), layout='constrained')
    for method, color, label, line in [('rs', ORANGE, 'Random search', '--'), ('bo', BLUE, 'BO', '-')]:
        q1, median, q3 = np.percentile(best[method], [25, 50, 75], axis=0)
        axes[0].fill_between(x, q1, q3, step='post', color=color, alpha=.13, linewidth=0)
        axes[0].step(x, median, where='post', color=color, lw=1.9, ls=line, label=label)
    search_axis(axes[0])
    axes[0].set_ylim(np.floor(min(b.min() for b in best.values())/5)*5-1, 90)
    axes[0].set_title('(a) Median best-so-far inner-CV accuracy and IQR', loc='left', fontsize=10)
    axes[0].legend(loc='lower right', frameon=False)
    for k in range(1, 6):
        pair = results[results.outer_fold == k].set_index('method')
        axes[1].plot([k, k+.20], [pair.loc['rs', 'outer_accuracy']*100,
                                  pair.loc['bo', 'outer_accuracy']*100], color='#8C959E', lw=1.2)
    for method, offset, color, label, marker in [
        ('fixed_reference', -.20, GREEN, 'Fixed reference', 's'),
        ('rs', 0, ORANGE, 'Random search', 'o'), ('bo', .20, BLUE, 'BO', 'D')]:
        subset = results[results.method == method].sort_values('outer_fold')
        axes[1].scatter(subset.outer_fold+offset, subset.outer_accuracy*100, s=39,
                        color=color, marker=marker, zorder=3,
                        label=f'{label} (mean {subset.outer_accuracy.mean()*100:.1f}%)')
    axes[1].set(xlim=(.5, 5.5), ylim=(70, 100), yticks=range(70, 101, 5),
                xticks=range(1, 6), xlabel='Outer fold', ylabel='Outer test accuracy (%)')
    axes[1].grid(axis='y', color='#DFE6EC', lw=.65)
    axes[1].set_title('(b) Outer test accuracy: paired BO and RS predictions', loc='left', fontsize=10)
    axes[1].legend(loc='lower right', frameon=False, fontsize=8)
    save(fig, 'fig5_2_nested_search_and_outer_accuracy.png')

    methods = {m['method']: m['pooled_outer_metrics'] for m in summary['methods']}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), layout='constrained')
    for ax, method, title in zip(axes, ['fixed_reference', 'rs', 'bo'], ['Fixed reference', 'Random search', 'BO']):
        cm = np.asarray(methods[method]['confusion_matrix'])
        assert cm.shape == (5, 5) and cm.sum() == 500 and np.all(cm.sum(axis=1) == 100)
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
        for row in range(5):
            for column in range(5):
                ax.text(column, row, str(cm[row, column]), ha='center', va='center',
                        color='white' if cm[row, column] >= 55 else '#243849', fontsize=9)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=list('ZONFS'),
               yticklabels=list('ZONFS'), xlabel='Predicted class', title=title)
        for spine in ax.spines.values(): spine.set_visible(False)
    axes[0].set_ylabel('True class')
    fig.colorbar(im, ax=axes, shrink=.84, label='Number of segments', ticks=[0, 25, 50, 75, 100])
    save(fig, 'fig5_3_nested_pooled_confusion_matrices.png')

    mappings = pd.read_csv(ROOT/'results/09_fixed_fold_segment_accuracy/repeated/per_seed.csv')
    fig, ax = plt.subplots(figsize=(7.15, 3.25), layout='constrained')
    names = ['uniform', 'normal', 'exponential', 'gamma']
    for seed, marker, offset in zip(range(42, 47), ['o', 's', 'D', '^', 'v'], np.linspace(-.13, .13, 5)):
        values = mappings[mappings.seed == seed].set_index('prior').loc[names, 'segment_accuracy']*100
        ax.scatter(np.arange(4)+offset, values, s=30, marker=marker, label=f'Seed {seed}')
    means = mappings.groupby('prior').segment_accuracy.mean().loc[names].to_numpy()*100
    for i, value in enumerate(means): ax.plot([i-.23, i+.23], [value, value], color='#243849', lw=1.5)
    ax.set(xticks=range(4), xticklabels=['Uniform', 'Truncated normal', 'Truncated exponential', 'Truncated Gamma'],
           ylim=(75, 95), yticks=range(75, 96, 5), ylabel='Holdout test accuracy (%)')
    ax.grid(axis='y', color='#DFE6EC', lw=.65)
    ax.legend(ncols=5, loc='upper center', frameon=False, fontsize=7.8)
    save(fig, 'fig5_4_exploratory_mapping_test_accuracy.png')

    print('Best inner CV at initialisation and search end:')
    for method in ['bo', 'rs']:
        print(method, 'at 16:', best[method][:, 15].tolist(), 'at 50:', best[method][:, -1].tolist())
    print('All PNG figures saved:', OUT)


if __name__ == '__main__':
    main()
