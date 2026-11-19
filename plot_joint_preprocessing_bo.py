"""Validate and summarize the completed 53-feature paired BO comparison."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_joint_preprocessing_bo import OUTPUT,SEEDS,BUDGET,summarize


def main():
    summarize()
    rows=[]; curves={'svm_only':[],'joint':[]}
    for seed in SEEDS:
        reference=None; folds_reference=None; initial_reference=None
        for method in curves:
            d=OUTPUT/f'seed_{seed}'/method
            r=json.loads((d/'final_result.json').read_text())
            h=pd.read_csv(d/'history.csv')
            split=json.loads((d/'split.json').read_text())
            folds=json.loads((d/'folds.json').read_text())
            assert len(h)==BUDGET and np.isclose(h.score.max(),r['best_cv_accuracy'])
            initial=h.loc[:15,['C','gamma']].to_numpy()
            if reference is None:reference=split;folds_reference=folds;initial_reference=initial
            else:
                assert split==reference and folds==folds_reference
                np.testing.assert_allclose(initial,initial_reference,rtol=1e-12)
            predictions=pd.read_csv(d/'test_predictions.csv')
            assert predictions.recording_id.nunique()==len(split['test_segment_ids'])
            assert set(predictions.recording_id)==set(split['test_segment_ids'])
            assert not set(predictions.recording_id)&set(split['train_segment_ids'])
            np.testing.assert_allclose((predictions.true_label==predictions.predicted_label).mean(),r['test_metrics']['accuracy'])
            rows.append(dict(seed=seed,method=method,cv=r['best_cv_accuracy'],test=r['test_metrics']['accuracy']))
            curves[method].append(h.score.cummax().to_numpy()*100)
    frame=pd.DataFrame(rows)
    paired=frame.pivot(index='seed',columns='method',values=['cv','test'])
    paired[('cv','joint_minus_svm')]=paired['cv']['joint']-paired['cv']['svm_only']
    paired[('test','joint_minus_svm')]=paired['test']['joint']-paired['test']['svm_only']
    paired.to_csv(OUTPUT/'paired_differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    labels={'svm_only':'Fixed preprocessing + BO-SVM','joint':'Joint BO'}
    for method,values in curves.items():
        a=np.array(values); mean=a.mean(axis=0);sd=a.std(axis=0,ddof=1);x=np.arange(1,BUDGET+1)
        axes[0].plot(x,mean,label=labels[method]);axes[0].fill_between(x,mean-sd,mean+sd,alpha=.15)
    axes[0].set(xlabel='Candidate evaluations',ylabel='Best recording CV accuracy (%)')
    axes[0].legend(fontsize=8,frameon=False)
    for seed in SEEDS:
        axes[1].plot([0,1],paired.loc[seed,'test'][['svm_only','joint']].to_numpy()*100,marker='o',label=f'Seed {seed}')
    axes[1].set_xticks([0,1],['Fixed preprocessing','Joint BO'])
    axes[1].set_ylabel('Outer-test recording accuracy (%)')
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'comparison.png',dpi=200);fig.savefig(OUTPUT/'comparison.pdf')
    print('All 10 runs validated: histories, paired splits, folds, initial SVM candidates and test predictions.')
    print(pd.read_csv(OUTPUT/'summary.csv').to_string(index=False))


if __name__=='__main__':
    import argparse
    import time
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true',help='Wait for all searches to finish before validating')
    args=parser.parse_args()
    if args.wait:
        deadline=time.monotonic()+24*3600
        expected=[OUTPUT/f'seed_{seed}'/method/name for seed in SEEDS
                  for method in ('svm_only','joint')
                  for name in ('final_result.json','test_predictions.csv')]
        while not all(path.exists() for path in expected):
            if time.monotonic()>deadline:raise TimeoutError('Searches incomplete after 24 hours')
            time.sleep(15)
    main()
