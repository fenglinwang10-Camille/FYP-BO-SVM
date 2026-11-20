"""Screen the existing 23 basic features on the same training-only pilot protocol."""

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from data_loader import _infer_class_from_path
from features import extract_features_by_setting
from grouped_bo_optimizer import _make_model, _majority_vote_by_segment, _segment_with_groups, _metrics
from preprocessing import bandpass_filter_signals

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'results/12_training_feature_pilot/catch24_outer_fold_1'
OUT = ROOT / 'results/12_training_feature_pilot/basic23_outer_fold_1'


def save_json(name, value):
    target = OUT/name
    temporary = target.with_name(target.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def main():
    reference = json.loads((REFERENCE/'protocol.json').read_text())
    assert reference['outer_test_access'] is False
    for name in ['features.py','preprocessing.py','data_loader.py','grouped_bo_optimizer.py']:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == reference['sources'][name]
    ids = np.array(reference['train_segment_ids'])
    assert len(ids) == 400
    # The saved reference manifest names only the 400 original training files.
    paths = [ROOT/'Dataset'/name for name in reference['train_files']]
    signals = [np.loadtxt(path).reshape(-1) for path in paths]
    labels = np.array([_infer_class_from_path(path,ROOT/'Dataset') for path in paths])
    assert np.array_equal(np.bincount(labels),[80]*5)
    digest = hashlib.sha256()
    for signal in signals:
        digest.update(signal.astype('<f8').tobytes())
    assert digest.hexdigest() == reference['training_data_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Existing pilot is preserved; do not overwrite its protocol.')
    protocol = dict(reference)
    for key in ['source_repository','upstream_definition','citation','preprocessing_seconds']:
        protocol.pop(key,None)
    protocol.update(experiment='basic23_training_only_feature_screen_v1',
                    feature_definition='original: 5 absolute powers + 5 relative powers + 13 statistics',
                    reference_protocol_sha256=hashlib.sha256((REFERENCE/'protocol.json').read_bytes()).hexdigest(),
                    pilot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json('protocol.json',protocol)
    save_json('status.json',dict(stage='extracting',outer_test_access=False))
    prep=reference['fixed_preprocessing']
    started=perf_counter()
    filtered=bandpass_filter_signals(signals,reference['sampling_frequency'],prep['f_low'],prep['f_high'])
    windows,y,groups=_segment_with_groups(filtered,labels,ids,prep['window_size'])
    preprocessing_seconds=perf_counter()-started
    started=perf_counter()
    X=extract_features_by_setting(windows,reference['sampling_frequency'],'original')
    extraction_seconds=perf_counter()-started
    assert X.shape==(4400,23) and np.isfinite(X).all()
    cached=joblib.load(REFERENCE/'original_55_training_features.joblib')
    np.testing.assert_array_equal(y,cached['y'])
    np.testing.assert_array_equal(groups,cached['groups'])
    # In the 55-column layout six ratios separate powers from the statistics.
    np.testing.assert_allclose(X,np.concatenate([cached['X'][:,:10],cached['X'][:,16:29]],axis=1),rtol=1e-12,atol=1e-12)
    del cached
    print(f'basic23: {X.shape}, extraction={extraction_seconds:.2f}s; verified exact baseline subset',flush=True)
    rows=[]
    for number,params in enumerate(reference['candidates'],1):
        started=perf_counter()
        accuracies,recalls=[],[]
        for fold in reference['inner_folds']:
            fit_ids,validation_ids=set(fold['train_segment_ids']),set(fold['validation_segment_ids'])
            assert not fit_ids & validation_ids and fit_ids | validation_ids == set(ids)
            fit=np.isin(groups,list(fit_ids))
            validation=np.isin(groups,list(validation_ids))
            model=_make_model(params)
            model.fit(X[fit],y[fit])
            truth,prediction=_majority_vote_by_segment(model.predict(X[validation]),y[validation],groups[validation])
            assert len(truth)==80
            metrics=_metrics(truth,prediction)
            accuracies.append(metrics['accuracy'])
            recalls.append([metrics['per_class_recall'][c] for c in 'ZONFS'])
        row=dict(feature_setting='basic23',candidate=number,**params,
                 mean_inner_cv_accuracy=float(np.mean(accuracies)),fold_accuracy=accuracies,
                 candidate_seconds=perf_counter()-started)
        row.update({f'recall_{c}':float(v) for c,v in zip('ZONFS',np.mean(recalls,axis=0))})
        rows.append(row)
        save_json('history.json',rows)
        pd.DataFrame(rows).to_csv(OUT/'history.csv',index=False)
        save_json('status.json',dict(stage='searching',candidate=number,total=12,outer_test_access=False))
        print(f'{number}/12 C={params["C"]:g}, gamma={params["gamma"]:g}, CV={row["mean_inner_cv_accuracy"]:.4f}, {row["candidate_seconds"]:.1f}s',flush=True)
    best=max(rows,key=lambda row:row['mean_inner_cv_accuracy'])
    result=dict(best,feature_columns=23,extraction_seconds=extraction_seconds,
                preprocessing_seconds=preprocessing_seconds,search_seconds=sum(r['candidate_seconds'] for r in rows))
    save_json('summary.json',dict(results=[result],outer_test_access=False))
    pd.DataFrame([result]).to_csv(OUT/'summary.csv',index=False)
    comparisons=json.loads((REFERENCE/'summary.json').read_text())['results']+[result]
    pd.DataFrame(comparisons).to_csv(OUT/'feature_comparison.csv',index=False)
    save_json('status.json',dict(stage='complete',outer_test_access=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
