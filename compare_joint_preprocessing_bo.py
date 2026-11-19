"""Paired 53-feature BO: fixed preprocessing vs joint optimisation.

Separate checkpointed runs; never replace the earlier 55-feature results.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Real
from skopt.utils import cook_estimator, normalize_dimensions
from analyze_grouped_features import REFERENCE_PARAMS, _saved_training_folds
from compare_feature_deduplication import KEEP, evaluate
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (DATASET_PATH,FIXED_FOLD_DIR,_build_feature_matrix,
    _select_segments,_make_model,_majority_vote_by_segment,_metrics,_json_metrics)

OUTPUT=FIXED_FOLD_DIR/'joint_vs_svm_only_53'
SEEDS=[42,43,44,45,46]
BUDGET=50
INITIAL=16


def make_space(method):
    svm=[Real(.1,100,prior='log-uniform',name='C'),Real(1e-4,1,prior='log-uniform',name='gamma')]
    return ([Real(.5,5,name='f_low'),Real(30,60,name='f_high'),Real(1,5,name='window_size')]
            if method=='joint' else [])+svm


def parameters(point,method):
    return {**REFERENCE_PARAMS,**dict(zip([d.name for d in make_space(method)],map(float,point)))}


def atomic_dump(state,path):
    temporary=path.with_suffix('.tmp')
    joblib.dump(state,temporary)
    os.replace(temporary,path)


def run_seed(seed):
    raw,labels=load_bonn_dataset(DATASET_PATH,classification='five-class')
    labels=np.asarray(labels,dtype=int)
    ids,folds=_saved_training_folds(seed,labels)
    train=_select_segments(raw,labels,ids)
    saved=FIXED_FOLD_DIR/f'seed_{seed}'/'uniform'
    split=json.loads((saved/'grouped_split_manifest.json').read_text())
    test_ids=np.asarray(split['test_segment_ids'],dtype=int)
    assert not set(ids)&set(test_ids)
    # Same initial C/gamma in both arms; first configuration is the reference.
    rng=np.random.default_rng(seed)
    svm_initial=[[10.,.01]]+[[float(np.exp(rng.uniform(np.log(.1),np.log(100)))),
        float(np.exp(rng.uniform(np.log(1e-4),0)))] for _ in range(INITIAL-1)]
    pre_rng=np.random.default_rng(seed+10000)
    pre_initial=[[1.,40.,5.]]+[[float(pre_rng.uniform(.5,5)),float(pre_rng.uniform(30,60)),
                                float(pre_rng.uniform(1,5))] for _ in range(INITIAL-1)]
    for method in ('svm_only','joint'):
        directory=OUTPUT/f'seed_{seed}'/method
        directory.mkdir(parents=True,exist_ok=True)
        final=directory/'final_result.json'
        if final.exists():
            print(f'seed={seed} {method}: already completed',flush=True)
            continue
        (directory/'split.json').write_text(json.dumps(split,indent=2))
        (directory/'folds.json').write_text(json.dumps(folds,indent=2))
        checkpoint=directory/'checkpoint.joblib'
        if checkpoint.exists():
            state=joblib.load(checkpoint)
            optimizer,history=state['optimizer'],state['history']
        else:
            space=make_space(method)
            gp=cook_estimator('GP',space=normalize_dimensions(space),noise='gaussian',random_state=seed)
            optimizer=Optimizer(space,base_estimator=gp,n_initial_points=INITIAL,
                                acq_func='EI',random_state=seed)
            history=[]
        assert len(history)==len(optimizer.Xi)<=BUDGET
        # Only a single-entry cache is needed; joint search may visit many windows.
        cache_key=None; cache=None
        def matrix(params):
            nonlocal cache_key,cache
            key=tuple(params[k] for k in ('f_low','f_high','window_size'))
            if key!=cache_key:
                X,y,g=_build_feature_matrix(*train,params)
                cache=(X[:,KEEP],y,g); cache_key=key
                assert cache[0].shape[1]==53
            return cache
        for iteration in range(len(history),BUDGET):
            started=perf_counter()
            if iteration<INITIAL:
                point=(pre_initial[iteration] if method=='joint' else [])+svm_initial[iteration]
            else:
                point=optimizer.ask()
            params=parameters(point,method)
            X,y,g=matrix(params)
            score,metrics,_=evaluate(X,y,g,folds,params)
            evaluated=perf_counter()
            optimizer.tell(point,-score)
            history.append(dict(iteration=iteration+1,**params,score=score,
                cv_macro_f1=metrics['macro_f1'],evaluation_seconds=evaluated-started,
                total_step_seconds=perf_counter()-started))
            atomic_dump(dict(optimizer=optimizer,history=history),checkpoint)
            pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
            print(f'seed={seed} {method} {iteration+1}/{BUDGET} CV={score:.4f}',flush=True)
        # Recover CSV if interrupted after checkpoint replacement.
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        best=max(history,key=lambda r:r['score'])
        params={k:best[k] for k in REFERENCE_PARAMS}
        X,y,g=matrix(params)
        model=_make_model(params); model.fit(X,y)
        test=_select_segments(raw,labels,test_ids)
        Xt,yt,gt=_build_feature_matrix(*test,params); Xt=Xt[:,KEEP]
        predictions=model.predict(Xt)
        truth,pred=_majority_vote_by_segment(predictions,yt,gt)
        metrics=_metrics(truth,pred)
        result=dict(seed=seed,method=method,n_features=53,n_calls=BUDGET,n_initial=INITIAL,
            parameters=params,best_cv_accuracy=best['score'],best_iteration=best['iteration'],
            test_metrics=_json_metrics(metrics),test_evaluations=1,
            search_seconds=sum(r['total_step_seconds'] for r in history),
            limitation='Development-exposed overlapping outer splits; not new independent confirmation.')
        temporary=final.with_suffix('.tmp'); temporary.write_text(json.dumps(result,indent=2)); os.replace(temporary,final)
        pd.DataFrame(dict(recording_id=np.unique(gt),true_label=truth,predicted_label=pred)).to_csv(directory/'test_predictions.csv',index=False)
        print(f'COMPLETE seed={seed} {method} CV={best["score"]:.4f} test={metrics["accuracy"]:.4f}',flush=True)


def summarize():
    rows=[]
    for seed in SEEDS:
        for method in ('svm_only','joint'):
            path=OUTPUT/f'seed_{seed}'/method/'final_result.json'
            if path.exists():
                r=json.loads(path.read_text())
                rows.append(dict(seed=seed,method=method,cv_accuracy=r['best_cv_accuracy'],
                    test_accuracy=r['test_metrics']['accuracy'],test_macro_f1=r['test_metrics']['macro_f1'],
                    search_seconds=r['search_seconds']))
    frame=pd.DataFrame(rows)
    if frame.empty:return
    frame.to_csv(OUTPUT/'per_seed.csv',index=False)
    frame.groupby('method').agg(n=('seed','size'),cv_mean=('cv_accuracy','mean'),cv_sd=('cv_accuracy','std'),
        test_mean=('test_accuracy','mean'),test_sd=('test_accuracy','std')).to_csv(OUTPUT/'summary.csv')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:summarize();raise SystemExit
    protocol=dict(seeds=SEEDS,budget=BUDGET,initial=INITIAL,features=53,wavelet='db4',level=3,
        fixed_preprocessing={k:REFERENCE_PARAMS[k] for k in ('f_low','f_high','window_size')},
        gp='Constant * ARD Matern 5/2 + Gaussian observation noise; normalized targets',acquisition='EI',
        initial_design='shared initial C/gamma; first full configuration shared; remaining joint preprocessing uniform',
        metric='Mean fold record accuracy after majority vote',test_policy='one selected configuration per run',
        interpretation='Paired development-stage outer evaluation; previous test exposure and overlapping splits acknowledged',
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                [__file__,'features.py','preprocessing.py','grouped_bo_optimizer.py',
                 'compare_feature_deduplication.py','analyze_grouped_features.py']})
    path=OUTPUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_seed,SEEDS):summarize()
    summarize()
