"""Draw outer-test accuracy and pooled class-recall comparisons as PNGs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42'
OUT = ROOT / 'figures/chapter5'
METHODS = [('fixed_reference', 'Fixed reference', '#42835A'),
           ('rs', 'RS', '#AD4E24'), ('bo', 'BO', '#096398')]


def plot_accuracy(results):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True,
                             layout='constrained', width_ratios=[1, 1.12])
    labels = []
    for pos, (method, name, color) in enumerate(METHODS):
        scores = results[results.method == method].sort_values('outer_fold').outer_accuracy.to_numpy() * 100
        assert len(scores) == 5
        mean, sd = scores.mean(), scores.std(ddof=1)
        axes[0].scatter(pos + np.linspace(-.12, .12, 5), scores, color=color,
                        s=23, alpha=.70, edgecolors='white', linewidths=.4, zorder=3)
        axes[0].errorbar(pos, mean, yerr=sd, fmt='D', color='#202A33',
                         markersize=5, capsize=5, linewidth=1.3, zorder=4)
        labels.append(f'{name}\n{mean:.1f} ± {sd:.2f}')
        print(f'{name}: mean={mean:.1f}%, sample SD={sd:.5f} pp')
    axes[0].set(xticks=range(3), xticklabels=labels, xlim=(-.5, 2.5),
                ylabel='Outer test accuracy (%)')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].set_title('(a) Five-fold accuracy: mean ± SD', loc='left', fontsize=9)
    for fold in range(1, 6):
        paired = results[results.outer_fold == fold].set_index('method')
        scores = paired.loc[['rs', 'bo'], 'outer_accuracy'].to_numpy() * 100
        axes[1].plot([0, 1], scores, color='#8C959E', linewidth=1.2, zorder=2)
        axes[1].scatter(0, scores[0], marker='o', color=METHODS[1][2], s=28, zorder=3)
        axes[1].scatter(1, scores[1], marker='D', color=METHODS[2][2], s=28, zorder=3)
        axes[1].text(1.08, scores[1], f'Fold {fold} ({scores[1]-scores[0]:+.0f} pp)',
                      va='center', fontsize=7.5)
    axes[1].set(xticks=[0, 1], xticklabels=['RS', 'BO'], xlim=(-.25, 1.80))
    axes[1].set_title('(b) Paired outer-fold accuracy', loc='left', fontsize=9)
    for ax in axes:
        ax.set_ylim(70, 100)
        ax.set_yticks(range(70, 101, 5))
        ax.grid(axis='y', color='#DFE6EC', linewidth=.65)
        ax.set_axisbelow(True)
    target = OUT / 'fig5_2_nested_search_and_outer_accuracy.png'
    fig.savefig(target, dpi=320, facecolor='white')
    plt.close(fig)
    print(target)


def plot_recall(summary):
    metrics = {m['method']: m['pooled_outer_metrics'] for m in summary['methods']}
    fig, ax = plt.subplots(figsize=(7.15, 3.0), layout='constrained')
    x = np.arange(5)
    for offset, (method, name, color) in zip([-.24, 0, .24], METHODS):
        cm = np.asarray(metrics[method]['confusion_matrix'])
        assert cm.sum() == 500 and np.all(cm.sum(axis=1) == 100)
        recall = np.diag(cm) / cm.sum(axis=1) * 100
        bars = ax.bar(x + offset, recall, width=.22, color=color, label=name, zorder=3)
        ax.bar_label(bars, labels=[f'{v:.0f}' for v in recall], padding=3, fontsize=8)
        print(f'{name} pooled recall Z/O/N/F/S: {recall.tolist()}')
    ax.set(xticks=x, xticklabels=list('ZONFS'), xlabel='True class',
           ylabel='Pooled outer-test recall (%)', ylim=(0, 113), yticks=range(0, 101, 20))
    ax.grid(axis='y', color='#DFE6EC', linewidth=.65)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncols=3, frameon=False)
    target = OUT / 'fig5_3_nested_class_recall_comparison.png'
    fig.savefig(target, dpi=320, facecolor='white')
    plt.close(fig)
    print(target)


def plot_sensitivity_specificity(summary):
    """Compute one-versus-rest metrics from pooled outer-test confusion counts."""
    metrics = {m['method']: m['pooled_outer_metrics'] for m in summary['methods']}
    rows, totals = [], []
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.0), layout='constrained')
    x = np.arange(5)
    for offset, (method, name, color) in zip([-.24, 0, .24], METHODS):
        cm = np.asarray(metrics[method]['confusion_matrix'])
        assert cm.sum() == 500 and np.all(cm.sum(axis=1) == 100)
        tp = np.diag(cm)
        fn, fp = cm.sum(axis=1)-tp, cm.sum(axis=0)-tp
        tn = cm.sum()-tp-fn-fp
        sensitivity = 100 * tp / (tp+fn)
        specificity = 100 * tn / (tn+fp)
        for i, label in enumerate('ZONFS'):
            rows.append(dict(method=method, class_label=label, tp=int(tp[i]),
                             fn=int(fn[i]), fp=int(fp[i]), tn=int(tn[i]),
                             sensitivity_percent=sensitivity[i],
                             recall_percent=sensitivity[i], specificity_percent=specificity[i]))
        totals.append(dict(method=method, macro_recall_percent=sensitivity.mean(),
                           macro_specificity_percent=specificity.mean(),
                           seizure_sensitivity_percent=sensitivity[4],
                           seizure_specificity_percent=specificity[4]))
        for ax, values in zip(axes, [sensitivity, specificity]):
            bars = ax.bar(x+offset, values, width=.22, color=color, label=name, zorder=3)
            ax.bar_label(bars, labels=[f'{v:g}' for v in np.round(values, 2)],
                         padding=3, fontsize=7)
    for ax, title, ylabel in zip(axes,
        ['(a) Class sensitivity (recall)', '(b) Class specificity: one versus rest'],
        ['Sensitivity / recall (%)', 'Specificity (%)']):
        ax.set(xticks=x, xticklabels=list('ZONFS'), ylim=(0, 115), yticks=range(0,101,20),
               ylabel=ylabel)
        ax.set_title(title, loc='left', fontsize=10)
        ax.grid(axis='y', color='#DFE6EC', linewidth=.65)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1,1.23), ncols=3, frameon=False, fontsize=8)
    axes[1].set_xlabel('Class')
    target = OUT / 'fig5_3_nested_sensitivity_specificity.png'
    fig.savefig(target, dpi=320, facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'nested_outer_class_metrics.csv', index=False)
    pd.DataFrame(totals).to_csv(OUT/'nested_outer_metric_summary.csv', index=False)
    print(pd.DataFrame(totals).to_string(index=False))
    print(target)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        plot_accuracy(pd.read_csv(RUN / 'outer_fold_results.csv'))
        plot_recall(json.loads((RUN / 'summary.json').read_text()))
        plot_sensitivity_specificity(json.loads((RUN / 'summary.json').read_text()))


if __name__ == '__main__':
    main()
