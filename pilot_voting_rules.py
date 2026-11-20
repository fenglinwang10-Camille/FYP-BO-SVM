"""Compare recording aggregation rules on cached training-only 55-D features.

Each SVM is fitted once per candidate and inner fold. Its same window outputs
are evaluated under all three rules. No outer-test file is opened.
"""

import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from grouped_bo_optimizer import _make_model, _metrics

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT/'results/12_training_feature_pilot/catch24_outer_fold_1'
OUT=ROOT/'results/12_training_feature_pilot/voting_rules_outer_fold_1'
RULES=('majority_lowest_index','majority_score_tiebreak','mean_decision_score')


def save_json(name,value):
    target=OUT/name
    temp=target.with_name(target.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(target)


def aggregate(predictions,scores):
    counts=np.bincount(predictions,minlength=5)
    tied=np.flatnonzero(counts==counts.max())
    average=scores.mean(axis=0)
    return (int(tied[0]), int(tied[np.argmax(average[tied])]),
            int(np.argmax(average))), bool(len(tied)>1)


def check_aggregation():
    # A 2-2-1 multiclass tie: score tie-breaking must stay within tied classes.
    scores=np.tile([0.,1.,0.,3.,100.],(5,1))
    predictions,tie=aggregate(np.array([1,1,3,3,4]),scores)
    assert predictions==(1,3,4) and tie
    # With a unique vote winner, the tie-only rule must not alter the result.
    predictions,tie=aggregate(np.array([1,1,1,3,4]),scores)
    assert predictions==(1,1,4) and not tie


def main():
    check_aggregation()
    reference=json.loads((REFERENCE/'protocol.json').read_text())
    assert reference['outer_test_access'] is False
    for name in ['features.py','preprocessing.py','data_loader.py','grouped_bo_optimizer.py']:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==reference['sources'][name]
    cached=joblib.load(REFERENCE/'original_55_training_features.joblib')
    X,y,groups=cached['X'],cached['y'],cached['groups']
    assert X.shape==(4400,55) and np.isfinite(X).all()
    assert set(groups)==set(reference['train_segment_ids']) and len(set(groups))==400
    original_history=json.loads((REFERENCE/'history.json').read_text())
    original_history={r['candidate']:r for r in original_history if r['feature_setting']=='original_55'}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Existing pilot results are preserved; use a new output for another run.')
    protocol=dict(reference)
    for key in ['source_repository','upstream_definition','citation','preprocessing_seconds']:
        protocol.pop(key,None)
    protocol.update(experiment='training_only_voting_rules_v1',feature_setting='original_55',
                    feature_cache_sha256=hashlib.sha256((REFERENCE/'original_55_training_features.joblib').read_bytes()).hexdigest(),
                    pilot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    aggregation_rules=list(RULES),
                    decision_scores='SVC OVR scores in class order Z,O,N,F,S; not probabilities',
                    tiebreak='Maximum mean OVR score among vote-tied classes only; exact score ties choose lowest index',
                    shared_fitting='One fitted SVM per candidate/inner fold, reused for all aggregation rules')
    save_json('protocol.json',protocol)
    all_rows,all_predictions=[],[]
    for number,params in enumerate(reference['candidates'],1):
        started=perf_counter()
        fold_accuracies={rule:[] for rule in RULES}
        candidate_rows=[]
        for fold in reference['inner_folds']:
            fit_ids,validation_ids=set(fold['train_segment_ids']),set(fold['validation_segment_ids'])
            assert not fit_ids & validation_ids and fit_ids | validation_ids==set(groups)
            fit=np.isin(groups,list(fit_ids))
            validation=np.isin(groups,list(validation_ids))
            model=_make_model(params)
            model.fit(X[fit],y[fit])
            assert model.classes_.tolist()==list(range(5))
            window_predictions=model.predict(X[validation])
            scores=model.decision_function(X[validation])
            assert scores.shape==(int(validation.sum()),5) and np.isfinite(scores).all()
            truths,predictions=[],[]
            for group in sorted(validation_ids):
                mask=groups[validation]==group
                truth=np.unique(y[validation][mask])
                assert len(truth)==1
                outputs,tie=aggregate(window_predictions[mask],scores[mask])
                truths.append(int(truth[0]))
                predictions.append(outputs)
                candidate_rows.append(dict(candidate=number,inner_fold=fold['fold'],segment_id=int(group),
                                           y_true=int(truth[0]),vote_tie=tie,
                                           **dict(zip(RULES,outputs))))
            assert len(truths)==80
            predictions=np.asarray(predictions)
            for i,rule in enumerate(RULES):
                fold_accuracies[rule].append(float(np.mean(predictions[:,i]==truths)))
        assert len(candidate_rows)==400 and len({r['segment_id'] for r in candidate_rows})==400
        original=original_history[number]
        np.testing.assert_allclose(fold_accuracies[RULES[0]],original['fold_accuracy'],atol=1e-12,rtol=0)
        for rule in RULES:
            true=np.array([r['y_true'] for r in candidate_rows])
            predicted=np.array([r[rule] for r in candidate_rows])
            metrics=_metrics(true,predicted)
            base=np.array([r[RULES[0]] for r in candidate_rows])
            row=dict(rule=rule,candidate=number,**params,mean_inner_cv_accuracy=float(np.mean(fold_accuracies[rule])),
                     fold_accuracy=fold_accuracies[rule],vote_tie_count=sum(r['vote_tie'] for r in candidate_rows),
                     changed_vs_majority=int(np.sum(predicted!=base)),
                     corrected_vs_majority=int(np.sum((predicted==true)&(base!=true))),
                     harmed_vs_majority=int(np.sum((predicted!=true)&(base==true))),
                     shared_fit_and_prediction_seconds=perf_counter()-started)
            row.update({f'recall_{c}':float(metrics['per_class_recall'][c]) for c in 'ZONFS'})
            all_rows.append(row)
        all_predictions.extend(candidate_rows)
        save_json('history.json',all_rows)
        pd.DataFrame(all_rows).to_csv(OUT/'history.csv',index=False)
        save_json('status.json',dict(stage='running',candidate=number,total=12,outer_test_access=False))
        scores_text=', '.join(f'{rule}={100*np.mean(fold_accuracies[rule]):.2f}%' for rule in RULES)
        print(f'{number}/12 C={params["C"]:g}, gamma={params["gamma"]:g}: {scores_text}',flush=True)
    pd.DataFrame(all_predictions).to_csv(OUT/'inner_recording_predictions.csv',index=False)
    best=[max([r for r in all_rows if r['rule']==rule],key=lambda r:r['mean_inner_cv_accuracy']) for rule in RULES]
    # Also show all rules at the original baseline's chosen configuration.
    baseline=best[0]['candidate']
    matched=[r for r in all_rows if r['candidate']==baseline]
    save_json('summary.json',dict(best_per_rule=best,at_baseline_selected_candidate=matched,
                                   outer_test_access=False,baseline_reproduced_all_12_candidates=True))
    pd.DataFrame(best).to_csv(OUT/'best_per_rule.csv',index=False)
    pd.DataFrame(matched).to_csv(OUT/'same_configuration_comparison.csv',index=False)
    save_json('status.json',dict(stage='complete',outer_test_access=False))
    print(json.dumps(best,indent=2),flush=True)


if __name__=='__main__':
    main()
