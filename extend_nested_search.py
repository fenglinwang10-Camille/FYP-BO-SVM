"""Continue saved 50-evaluation score-tiebreak searches, training CV only.

The original checkpoint is verified and copied into a new protocol namespace.
The fitted GP, RNG state, candidate order and scores are preserved. No outer
test recordings or outer-test predictions are opened, and none are produced.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from data_loader import _infer_class_from_path
import nested_bo_rs as base
from nested_score_tiebreak import cross_validate

ROOT=Path(__file__).resolve().parent
DEFAULT_SOURCE=ROOT/'results/13_nested_score_tiebreak/split_42_inner_1729/search_42'
DEFAULT_OUTPUT=ROOT/'results/14_training_search_extension_100'


def verify_and_fork(source_state,old_protocol,new_protocol,split):
    if source_state['signature']!=base._digest({'protocol':old_protocol,'split':split}):
        raise ValueError('Source checkpoint signature does not match its frozen protocol')
    budget=old_protocol['evaluations_per_method']
    initial=old_protocol['shared_initial_evaluations']
    if source_state['budget']!=budget or len(source_state['shared'])!=initial:
        raise ValueError('Source initial history is incomplete')
    for method in ['bo','rs']:
        if len(source_state[method])!=budget-initial:
            raise ValueError('Source search is incomplete')
        rows=source_state['shared']+source_state[method]
        if [r['iteration'] for r in rows]!=list(range(1,budget+1)):
            raise ValueError('Unexpected source candidate numbering')
    rows=source_state['shared']+source_state['bo']
    expected=[[row[name] for name in base.PARAMETERS] for row in rows]
    np.testing.assert_allclose(source_state['optimizer'].Xi,expected,rtol=0,atol=1e-12)
    np.testing.assert_allclose(source_state['optimizer'].yi,[-r['score'] for r in rows],rtol=0,atol=1e-12)
    state=copy.deepcopy(source_state)
    state['signature']=base._digest({'protocol':new_protocol,'split':split})
    state['budget']=new_protocol['evaluations_per_method']
    return state


def load_training(split,files):
    ids=np.array(split['train_segment_ids'])
    assert len(ids)==400 and not set(ids)&set(split['test_segment_ids'])
    selected=[files[i] for i in ids]
    signals=[np.loadtxt(path,dtype=float).reshape(-1) for path in selected]
    labels=np.array([_infer_class_from_path(path,ROOT/'Dataset') for path in selected])
    assert np.array_equal(np.bincount(labels),[80]*5)
    return signals,labels,ids


def summarize(output,protocol):
    rows=[]
    for fold in range(1,6):
        for method in ['bo','rs']:
            path=output/f'outer_fold_{fold}'/f'{method}_history.csv'
            if not path.exists():continue
            frame=pd.read_csv(path,float_precision='round_trip')
            best=np.maximum.accumulate(frame.score.to_numpy())
            improving=np.r_[True,np.diff(best)>1e-12]
            last=int(np.flatnonzero(improving)[-1]+1)
            rows.append(dict(outer_fold=fold,method=method,evaluations=len(frame),
                             best_at_50=float(best[49]),best_current=float(best[-1]),
                             gain_after_50_pp=float((best[-1]-best[49])*100),
                             last_improvement=last,evaluations_since_improvement=len(frame)-last))
    base._csv_save(output/'training_search_summary.csv',rows)
    complete=len(rows)==10 and all(r['evaluations']==protocol['evaluations_per_method'] for r in rows)
    if complete:
        base._json_save(output/'summary.json',dict(training_only=True,outer_test_access=False,
                                                  evaluations_per_method=protocol['evaluations_per_method'],results=rows))
    return complete


def run(args):
    source=Path(args.source_run).resolve()
    output=Path(args.output_dir).resolve()/f'split_42_inner_1729/search_42'
    if source==output or source in output.parents:
        raise ValueError('Extension must be saved outside the original experiment')
    old_protocol=json.loads((source/'protocol.json').read_text())
    if old_protocol['protocol']!='nested_recording_cv_shared_initialization_score_tiebreak_v1':
        raise ValueError('This extension requires the completed score-tiebreak source experiment')
    old_budget=old_protocol['evaluations_per_method']
    if args.budget<=old_budget:
        raise ValueError('New budget must exceed the completed source budget')
    for name,digest in old_protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source file changed: {name}')
    for package,expected in old_protocol['versions'].items():
        if version(package)!=expected:
            raise ValueError(f'Package version changed: {package}')
    splits=json.loads((source/'nested_folds.json').read_text())
    np.testing.assert_array_equal(base.random_points(old_protocol['search_seed'],args.budget)[:old_budget],
                                   base.random_points(old_protocol['search_seed'],old_budget))
    protocol=copy.deepcopy(old_protocol)
    protocol.update(protocol='training_only_saved_gp_search_extension_v1',
                    evaluations_per_method=args.budget,source_evaluations_per_method=old_budget,
                    source_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),
                    source_checkpoint_sha256={str(s['outer_fold']):hashlib.sha256((source/f"outer_fold_{s['outer_fold']}"/'search_state.joblib').read_bytes()).hexdigest() for s in splits},
                    outer_test_access=False,
                    purpose='Exploratory budget sensitivity of inner validation search, not a new generalisation estimate',
                    continuation='Preserve first 50 scores, GP and RNG state; continue both methods without reinitialisation')
    protocol['source_sha256']['extend_nested_search.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    dataset=ROOT/'Dataset'
    files=sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.txt','.csv','.asc'})
    files=[p for p in files if _infer_class_from_path(p,dataset) is not None]
    assert len(files)==500
    with base.run_lock(output):
        base._immutable_json(output/'protocol.json',protocol)
        base._immutable_json(output/'nested_folds.json',splits)
        base._json_save(output/'run_status.json',dict(status='running',pid=os.getpid(),training_only=True))
        try:
            for split in splits:
                if args.folds and split['outer_fold'] not in args.folds:continue
                directory=output/f"outer_fold_{split['outer_fold']}"
                directory.mkdir(exist_ok=True)
                base._immutable_json(directory/'fold_manifest.json',split)
                if not (directory/'search_state.joblib').exists():
                    source_state=joblib.load(source/directory.name/'search_state.joblib')
                    state=verify_and_fork(source_state,old_protocol,protocol,split)
                    base._save_state(directory,state)
                    del state,source_state
                train_data=load_training(split,files)
                with patch.object(base,'_cross_validate_training_segments',cross_validate):
                    state,complete=base.run_searches(train_data,split,protocol,directory,args.max_new_evaluations)
                # Verify the preserved history against the source after every continuation.
                original_state=joblib.load(source/directory.name/'search_state.joblib')
                assert state['shared']==original_state['shared']
                for method in ['bo','rs']:
                    assert state[method][:old_budget-16]==original_state[method]
                summarize(output,protocol)
                if not complete:
                    base._json_save(output/'run_status.json',dict(status='paused_at_evaluation_limit',pid=os.getpid(),training_only=True))
                    return
                selected=base.selected_configurations(state,protocol)
                selected.pop('fixed_reference')
                base._immutable_json(directory/'selected_training_configurations.json',selected)
                del state,original_state,train_data
            complete=summarize(output,protocol)
            base._json_save(output/'run_status.json',dict(status='complete' if complete else 'partial_folds_complete',pid=os.getpid(),training_only=True))
        except BaseException as error:
            base._json_save(output/'run_status.json',dict(status='failed',error=str(error),pid=os.getpid(),training_only=True))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--budget',type=int,default=100)
    parser.add_argument('--folds',type=int,nargs='+',choices=range(1,6))
    parser.add_argument('--max-new-evaluations',type=int)
    args=parser.parse_args()
    if args.max_new_evaluations is not None and args.max_new_evaluations<1:
        parser.error('--max-new-evaluations must be positive')
    run(args)
