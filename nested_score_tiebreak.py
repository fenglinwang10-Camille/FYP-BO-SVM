"""Supplementary nested BO/RS experiment with score-based recording tie breaks.

The completed original experiment is unchanged. A process-local adapter reuses
its split, search, checkpoint and outer-evaluation scheduling machinery. Only
recording vote ties are resolved differently, in both inner CV and outer tests.
"""

import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import argparse
import hashlib
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
import grouped_bo_optimizer as pipeline
import nested_bo_rs as original

DEFAULT_OUTPUT=original.PROJECT_DIR/'results/13_nested_score_tiebreak'
ORIGINAL_PROTOCOL=original.make_protocol


def vote_with_scores(predictions, scores, labels, groups):
    """Use the average OVR score only among recording vote-tied classes."""
    predictions=np.asarray(predictions,dtype=int)
    scores=np.asarray(scores,dtype=float)
    labels,groups=np.asarray(labels),np.asarray(groups)
    if scores.shape!=(len(predictions),5) or not np.isfinite(scores).all():
        raise ValueError('Expected finite OVR scores in Z,O,N,F,S class order')
    if not (len(labels)==len(groups)==len(predictions)):
        raise ValueError('Window output lengths differ')
    truth,voted,diagnostics=[],[],[]
    for group in np.unique(groups):
        mask=groups==group
        targets=np.unique(labels[mask])
        if len(targets)!=1:
            raise ValueError('One recording has inconsistent labels')
        counts=np.bincount(predictions[mask],minlength=5)
        tied=np.flatnonzero(counts==counts.max())
        base=int(tied[0])
        chosen=base if len(tied)==1 else int(tied[np.argmax(scores[mask].mean(axis=0)[tied])])
        truth.append(int(targets[0]))
        voted.append(chosen)
        diagnostics.append(dict(recording_id=int(group),vote_tie=bool(len(tied)>1),
                                lowest_index_prediction=base,changed_by_tiebreak=chosen!=base))
    return np.array(truth),np.array(voted),diagnostics


def cross_validate(signals,labels,segment_ids,params,objective_mode='segment_accuracy',
                   random_state=42,feature_cache=None,fixed_folds=None):
    if objective_mode!='segment_accuracy' or fixed_folds is None:
        raise ValueError('This adapter requires fixed recording folds and recording accuracy')
    key=tuple(float(params[name]) for name in ('f_low','f_high','window_size'))
    if feature_cache is not None and key in feature_cache:
        X,y,groups=feature_cache[key]
    else:
        X,y,groups=pipeline._build_feature_matrix(signals,labels,segment_ids,params)
        if feature_cache is not None:
            feature_cache[key]=(X,y,groups)
    assert set(groups)==set(segment_ids)
    window_metrics,recording_metrics,counts=[],[],[]
    validation_coverage=[]
    for fit_ids,valid_ids in fixed_folds:
        assert not set(fit_ids)&set(valid_ids)
        assert set(fit_ids)|set(valid_ids)==set(segment_ids)
        fit,valid=np.isin(groups,fit_ids),np.isin(groups,valid_ids)
        model=pipeline._make_model(params)
        model.fit(X[fit],y[fit])
        assert model.classes_.tolist()==list(range(5))
        predictions=model.predict(X[valid])
        scores=model.decision_function(X[valid])
        truth,voted,_=vote_with_scores(predictions,scores,y[valid],groups[valid])
        window_metrics.append(pipeline._metrics(y[valid],predictions))
        recording_metrics.append(pipeline._metrics(truth,voted))
        validation_coverage.extend(valid_ids)
        counts.append(dict(train=len(fit_ids),validation=len(valid_ids)))
    assert sorted(validation_coverage)==sorted(segment_ids)
    def average(metrics):
        return dict(accuracy=float(np.mean([m['accuracy'] for m in metrics])),
                    macro_recall=float(np.mean([m['macro_recall'] for m in metrics])),
                    macro_f1=float(np.mean([m['macro_f1'] for m in metrics])),
                    per_class_recall={c:float(np.mean([m['per_class_recall'][c] for m in metrics]))
                                      for c in pipeline.CLASS_NAMES})
    metrics=average(window_metrics)
    metrics['segment']=average(recording_metrics)
    return metrics['segment']['accuracy'],metrics,X.shape,counts


def evaluate_outer(train_data,test_data,params):
    started=perf_counter()
    X,y,groups=pipeline._build_feature_matrix(*train_data,params)
    model=pipeline._make_model(params)
    model.fit(X,y)
    assert model.classes_.tolist()==list(range(5))
    # The reused scheduler calls this only after both selections are frozen.
    test_X,test_y,test_groups=pipeline._build_feature_matrix(*test_data,params)
    assert not set(groups)&set(test_groups)
    predictions=model.predict(test_X)
    truth,voted,diagnostics=vote_with_scores(predictions,model.decision_function(test_X),test_y,test_groups)
    return dict(segment_level_test_metrics=pipeline._json_metrics(pipeline._metrics(truth,voted)),
                window_level_test_metrics=pipeline._json_metrics(pipeline._metrics(test_y,predictions)),
                outer_evaluation_seconds=perf_counter()-started,
                recording_predictions=[dict(recording_id=int(r),true_label=int(t),predicted_label=int(p))
                                       for r,t,p in zip(np.unique(test_groups),truth,voted)],
                voting_diagnostics=diagnostics,
                tied_recordings=sum(d['vote_tie'] for d in diagnostics),
                changed_by_tiebreak=sum(d['changed_by_tiebreak'] for d in diagnostics))


def make_protocol(signals,labels,args):
    protocol=ORIGINAL_PROTOCOL(signals,labels,args)
    protocol.update(protocol='nested_recording_cv_shared_initialization_score_tiebreak_v1',
                    vote_rule='majority vote; ties use maximum mean OVR decision score among vote-tied classes; exact score ties use lowest index',
                    score_semantics='SVC OVR decision scores are not calibrated probabilities',
                    exploratory=True,
                    development_note='Tie rule selected after a 12-configuration inner-CV pilot on outer fold 1 training recordings. Existing dataset had been inspected in earlier experiments; supplementary reanalysis, not untouched external validation.')
    protocol['source_sha256']['nested_score_tiebreak.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return protocol


def run(args):
    # Reject accidental reuse of either completed experiment's output root.
    if Path(args.output_dir).resolve() in {original.DEFAULT_OUTPUT.resolve(),
          (original.PROJECT_DIR/'results/11_nested_mapping_comparison').resolve()}:
        raise ValueError('Use a separate output directory for the new voting protocol')
    with patch.object(original,'make_protocol',make_protocol), \
         patch.object(original,'_cross_validate_training_segments',cross_validate), \
         patch.object(original,'evaluate_outer',evaluate_outer):
        original.run(args)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer-split-seed',type=int,default=42)
    parser.add_argument('--inner-split-seed',type=int,default=1729)
    parser.add_argument('--search-seed',type=int,default=42)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--folds',type=int,nargs='+',choices=range(1,6))
    parser.add_argument('--max-new-evaluations',type=int)
    args=parser.parse_args()
    if args.max_new_evaluations is not None and args.max_new_evaluations<1:
        parser.error('--max-new-evaluations must be positive')
    return args


if __name__=='__main__':
    run(parse_args())
