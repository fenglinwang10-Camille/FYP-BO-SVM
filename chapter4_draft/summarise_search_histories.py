"""Audit and summarise saved searches without running experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/09_fixed_fold_segment_accuracy'
OUT = ROOT / 'chapter4_draft/search_history_report'
SEEDS = list(range(42, 47))
MAPS = ['uniform', 'normal', 'exponential', 'gamma']
LABELS = ['Uniform', 'Truncated normal', 'Truncated exponential', 'Truncated Gamma']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def history(path):
    h = pd.read_csv(path)
    assert len(h) == 50 and h.iteration.tolist() == list(range(1, 51)), path
    assert (h.feature_columns == 55).all(), path
    assert np.allclose(h.score, h.validation_segment_accuracy), path
    assert h.score.between(0, 1).all(), path
    return np.maximum.accumulate(h.score.to_numpy()) * 100

def setup():
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    ax.set(xlabel='Candidate evaluations', ylabel='Best-so-far validation accuracy (%)',
           xlim=(1, 50), xticks=[1, 10, 16, 20, 30, 40, 50])
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax

def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    x = np.arange(1, 51)
    curves = {m: [] for m in MAPS}
    records, sources = [], {}
    def register(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in SEEDS:
        reference = BASE / f'seed_{seed}/uniform'
        split = read_json(reference / 'grouped_split_manifest.json')
        folds = read_json(reference / 'grouped_cv_folds.json')
        train, test = set(split['train_segment_ids']), set(split['test_segment_ids'])
        assert len(train) == 400 and len(test) == 100 and not train & test
        validation_ids = []
        for fold in folds:
            a, b = set(fold['train_segment_ids']), set(fold['validation_segment_ids'])
            assert not a & b and a | b == train
            validation_ids.extend(b)
        assert len(validation_ids) == 400 and set(validation_ids) == train
        for m in MAPS:
            d = BASE / f'seed_{seed}' / m
            s = read_json(d / 'grouped_split_manifest.json')
            assert s['train_segment_ids'] == split['train_segment_ids']
            assert s['test_segment_ids'] == split['test_segment_ids']
            assert read_json(d / 'grouped_cv_folds.json') == folds
            hpath = d / 'grouped_bo_history.csv'
            c = history(hpath)
            curves[m].append(c)
            result = read_json(d / 'grouped_bo_final_result.json')
            assert result['n_initial_points'] == 16
            assert result['objective_mode'] == 'segment_accuracy'
            assert np.isclose(c[-1], 100 * result['best_validation_score'])
            for name in ['grouped_bo_history.csv', 'grouped_bo_final_result.json',
                         'grouped_split_manifest.json', 'grouped_cv_folds.json']:
                register(d / name)
            row = dict(seed=seed, mapping=m, best_validation_pct=c[-1],
                       test_accuracy_pct=100*result['segment_level_test_metrics']['accuracy'])
            row.update({f'best_validation_at_{k}_pct': c[k-1] for k in [16,20,30,40,50]})
            records.append(row)
    stats, long = [], []
    for m in MAPS:
        curves[m] = np.asarray(curves[m])
        for i in range(50):
            stats.append(dict(mapping=m, evaluation=i+1, n=5,
                              mean_pct=curves[m][:,i].mean(),
                              sd_pp=curves[m][:,i].std(ddof=1)))
        for seed, c in zip(SEEDS, curves[m]):
            long.extend(dict(seed=seed,mapping=m,evaluation=i+1,best_validation_pct=v)
                        for i,v in enumerate(c))
    pd.DataFrame(stats).to_csv(OUT/'mapping_history_mean_sd.csv', index=False)
    pd.DataFrame(long).to_csv(OUT/'mapping_histories_per_seed.csv', index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'mapping_results_per_seed.csv', index=False)
    differences = []
    for seed in SEEDS:
        sub = frame[frame.seed == seed].set_index('mapping')
        for m in MAPS[1:]:
            row = dict(seed=seed,comparison=f'{m} minus uniform')
            row.update({f'{col}_difference_pp': sub.loc[m,col]-sub.loc['uniform',col]
                        for col in sub.columns})
            differences.append(row)
    diffs = pd.DataFrame(differences)
    diffs.to_csv(OUT/'paired_differences_vs_uniform.csv', index=False)
    diffs.drop(columns='seed').groupby('comparison').agg(['mean','std']).to_csv(
        OUT/'paired_difference_summary.csv')
    fig, ax = setup()
    for m, label, color in zip(MAPS, LABELS, COLORS):
        mean, sd = curves[m].mean(axis=0), curves[m].std(axis=0, ddof=1)
        ax.step(x,mean,where='post',label=label,color=color,lw=1.8)
        ax.fill_between(x,mean-sd,mean+sd,step='post',color=color,alpha=.10)
    ax.axvline(16,color='#777777',ls=':',lw=1)
    ax.set_title('Four mappings: five runs each (mean and sample SD)')
    ax.legend(frameon=False,loc='lower right',ncol=2)
    save(fig,'mapping_validation_history_5seeds')
    fig, ax = setup()
    c = curves['uniform']; mean=c.mean(axis=0); sd=c.std(axis=0,ddof=1)
    ax.step(x,mean,where='post',color=COLORS[0],label='Uniform BO: five runs')
    ax.fill_between(x,mean-sd,mean+sd,step='post',color=COLORS[0],alpha=.18)
    ax.axvline(16,color='#777777',ls=':',lw=1)
    ax.set_title('Uniform BO: mean and sample SD, seeds 42–46')
    ax.legend(frameon=False)
    save(fig,'uniform_bo_validation_history_5seeds')
    rs_path=BASE/'method_comparison_seed_42/grouped_random_search_history.csv'
    rs=history(rs_path); register(rs_path)
    fig,ax=setup()
    ax.step(x,c[0],where='post',label='BO (seed 42)',color=COLORS[0],lw=1.8)
    ax.step(x,rs,where='post',label='Random search (seed 42)',color=COLORS[1],lw=1.8)
    ax.set_title('Matched seed-42 comparison (one run per method)')
    ax.legend(frameon=False)
    save(fig,'bo_vs_random_seed42')
    pd.DataFrame({'evaluation':x,'bo_seed42_best_validation_pct':c[0],
                  'random_seed42_best_validation_pct':rs,
                  'bo_minus_random_pp':c[0]-rs}).to_csv(OUT/'baseline_seed42_history.csv',index=False)
    protocol=dict(seeds=SEEDS,main_features=55,supplementary_features=53,
        objective='Negative mean fold recording-level validation accuracy; majority vote',
        split='Stratified 80/20 original recordings; 400 training, 100 test',
        cv='Fixed five recording-level folds per seed; fold-local scaling and SVM',
        budget=50,bo_initial=16,bo_guided=34,acquisition='EI',
        preprocessing={'filter':'Butterworth order argument 4; SOS forward/backward',
                       'overlap':0,'wavelet':'db4 level 3'},
        bounds={'f_low':[.5,5],'f_high':[30,60],'window_seconds':[1,5],
                'C':[.1,100],'gamma':[.0001,1]},
        svm_distribution='log-uniform for C and gamma',
        mappings={'uniform':'linear','normal':'loc midpoint; scale width/4',
                  'exponential':'loc lower bound; scale width/3',
                  'gamma':'shape 2; loc lower bound; scale width/4'},
        truncation='Nonuniform distributions restricted to parameter bounds; u in [1e-6,1-1e-6]',
        baseline_status={'BO_seeds':SEEDS,'random_search_seeds':[42],
                         'missing_random_search_seeds':[43,44,45,46]},
        aggregation='Per-run cumulative maximum, then mean and sample SD (ddof=1); percentages',
        paired_difference='Nonuniform mapping minus uniform, same seed; percentage points',
        limits=['Seeds change partition and search randomness.',
                'Outer splits overlap; development exposure acknowledged.',
                '55-feature and 53-feature results are separate experiments.',
                'No five-run random-search summary is available.'],
        source_sha256=sources)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Saved',OUT)
    print('Mapping mean final validation:', {m:round(curves[m][:,-1].mean(),3) for m in MAPS})
    print('Random search seeds 43–46 missing: no five-run baseline comparison generated.')

if __name__=='__main__':
    main()
