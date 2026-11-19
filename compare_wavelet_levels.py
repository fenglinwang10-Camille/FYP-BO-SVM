"""Training-only paired db4 level-3/4 comparison; main BO settings unchanged."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from analyze_grouped_features import REFERENCE_PARAMS, _saved_training_folds
from compare_feature_deduplication import evaluate, REMOVED
from data_loader import load_bonn_dataset
from features import extract_psd_statistical_features, extract_features_by_setting, _wavelet_features
from grouped_bo_optimizer import DATASET_PATH, FIXED_FOLD_DIR, FS, _select_segments, _segment_with_groups
from preprocessing import bandpass_filter_signals

OUTPUT = FIXED_FOLD_DIR / 'wavelet_levels_tuned_cv'
SEEDS = (42,43,44,45,46)
BUDGET = 50


def assemble(base, windows, level):
    if pywt.dwt_max_level(windows.shape[1], pywt.Wavelet('db4').dec_len) < level:
        raise ValueError('Requested decomposition level is infeasible')
    wave = np.asarray([_wavelet_features(w, wavelet='db4', level=level, enhanced=True)
                       for w in windows])
    if wave.shape != (len(windows), 5*(level+1)):
        raise ValueError('Wavelet extraction failed or returned the wrong dimension')
    raw = np.hstack([base, wave])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite features')
    return raw[:, [i for i in range(raw.shape[1]) if i not in REMOVED]]


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=SEEDS, budget=BUDGET, wavelet='db4', levels=[3,4],
        preprocessing={k:REFERENCE_PARAMS[k] for k in ['f_low','f_high','window_size']},
        removed_indices=list(REMOVED), dimensions={'3':53,'4':58},
        search='reference C=10 gamma=.01 plus 49 identical paired log-uniform random candidates',
        C_bounds=[.1,100], gamma_bounds=[1e-4,1], test_evaluations=0,
        caveat='Selected training-CV scores reuse tuning folds; overlapping splits are not independent. Level changes also change feature dimension.',
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ['features.py','preprocessing.py','grouped_bo_optimizer.py',
                        'compare_feature_deduplication.py','analyze_grouped_features.py',__file__]})
    protocol = json.loads(json.dumps(protocol))
    path = OUTPUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol mismatch; do not mix runs')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Check the shortest permitted window as well as the current fixed window.
    assert pywt.dwt_max_level(round(FS), pywt.Wavelet('db4').dec_len) >= 4
    raw, labels = load_bonn_dataset(DATASET_PATH,classification='five-class')
    labels = np.asarray(labels,dtype=int)
    summaries=[]
    for seed in SEEDS:
        train_ids, folds = _saved_training_folds(seed,labels)
        signals,y_record,ids = _select_segments(raw,labels,train_ids)
        filtered = bandpass_filter_signals(signals,FS,REFERENCE_PARAMS['f_low'],REFERENCE_PARAMS['f_high'])
        windows,y,groups = _segment_with_groups(filtered,y_record,ids,REFERENCE_PARAMS['window_size'])
        base = extract_psd_statistical_features(windows,FS,use_wavelet=False,use_band_ratios=True)
        assert base.shape[1] == 35
        rng=np.random.default_rng(seed)
        candidates=[(10.,.01)]+[(float(np.exp(rng.uniform(np.log(.1),np.log(100)))),
                                float(np.exp(rng.uniform(np.log(1e-4),0)))) for _ in range(BUDGET-1)]
        for level in (3,4):
            X=assemble(base,windows,level)
            assert X.shape[1] == (53 if level==3 else 58)
            if level==3:
                original=extract_features_by_setting(windows[:4],FS,'nf_enhanced')
                np.testing.assert_array_equal(X[:4],original[:,[i for i in range(55) if i not in REMOVED]])
            history_path=OUTPUT/f'seed_{seed}_level_{level}_history.csv'
            history=pd.read_csv(history_path).to_dict('records') if history_path.exists() else []
            assert len(history)<=BUDGET
            for row,pair in zip(history,candidates):
                np.testing.assert_allclose([row['C'],row['gamma']],pair,rtol=1e-12)
            for index in range(len(history),BUDGET):
                C,gamma=candidates[index]
                start=perf_counter()
                score,metrics,_=evaluate(X,y,groups,folds,{**REFERENCE_PARAMS,'C':C,'gamma':gamma})
                history.append(dict(iteration=index+1,C=C,gamma=gamma,cv_accuracy=score,
                                    cv_macro_f1=metrics['macro_f1'],seconds=perf_counter()-start))
                pd.DataFrame(history).to_csv(history_path,index=False)
                if (index+1)%10==0: print(f'seed {seed} level {level}: {index+1}/50',flush=True)
            best=max(history,key=lambda r:r['cv_accuracy'])
            score,metrics,predictions=evaluate(X,y,groups,folds,{**REFERENCE_PARAMS,'C':best['C'],'gamma':best['gamma']})
            assert np.isclose(score,best['cv_accuracy'])
            assert {r['recording_id'] for r in predictions} == set(train_ids.tolist())
            pd.DataFrame(predictions).to_csv(OUTPUT/f'seed_{seed}_level_{level}_selected_cv_predictions.csv',index=False)
            summaries.append(dict(seed=seed,level=level,n_features=X.shape[1],**best,
                                  cv_recall_F=metrics['per_class_recall']['F']))
            pd.DataFrame(summaries).to_csv(OUTPUT/'per_seed.csv',index=False)
            print(f'Completed seed {seed} level {level}: best CV {score:.4f}',flush=True)
    frame=pd.DataFrame(summaries)
    summary=frame.groupby('level').agg(n=('seed','size'),accuracy_mean=('cv_accuracy','mean'),
        accuracy_sd=('cv_accuracy','std'),macro_f1_mean=('cv_macro_f1','mean'),F_recall_mean=('cv_recall_F','mean'))
    summary.to_csv(OUTPUT/'summary.csv')
    paired=frame.pivot(index='seed',columns='level',values='cv_accuracy')
    paired['difference_4_minus_3']=paired[4]-paired[3]
    paired.to_csv(OUTPUT/'paired_differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for seed in SEEDS:
        axes[0].plot([3,4],paired.loc[seed,[3,4]].to_numpy()*100,marker='o',label=f'Seed {seed}')
    axes[0].set_xticks([3,4],['db4 level 3 (53)','db4 level 4 (58)'])
    axes[0].set_ylabel('Selected training-CV accuracy (%)')
    axes[0].legend(frameon=False,fontsize=8)
    for level in (3,4):
        curves=[]
        for seed in SEEDS:
            h=pd.read_csv(OUTPUT/f'seed_{seed}_level_{level}_history.csv')
            assert len(h)==BUDGET
            other=pd.read_csv(OUTPUT/f'seed_{seed}_level_3_history.csv')
            np.testing.assert_array_equal(h[['C','gamma']],other[['C','gamma']])
            curves.append(h.cv_accuracy.cummax().to_numpy()*100)
        curves=np.array(curves); mean=curves.mean(axis=0); sd=curves.std(axis=0,ddof=1)
        x=np.arange(1,BUDGET+1)
        axes[1].plot(x,mean,label=f'Level {level}')
        axes[1].fill_between(x,mean-sd,mean+sd,alpha=.15)
    axes[1].set_xlabel('SVM configurations evaluated')
    axes[1].set_ylabel('Best training-CV accuracy so far (%)')
    axes[1].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'comparison.png',dpi=200)
    fig.savefig(OUTPUT/'comparison.pdf')
    print(summary.to_string(),flush=True)


if __name__=='__main__': run()
