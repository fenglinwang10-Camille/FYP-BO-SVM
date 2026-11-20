"""Complete the three mapped BO comparisons on the frozen nested folds.

The completed uniform experiment is validated and reused without fitting.
All new searches use the same bounds, feature pipeline, folds and BO settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Real

from data_loader import load_bonn_dataset
from grouped_bo_optimizer import DATASET_PATH, PROJECT_DIR, SEARCH_SPACE, _select_segments
from nested_bo_rs import (
    PARAMETERS, _csv_save, _digest, _evaluate_candidate, _immutable_json,
    _json_save, evaluate_outer, make_protocol, make_splits, random_points, run_lock,
)
from parameter_priors import PREPROCESSING_BOUNDS, transform_preprocessing_parameters

MAPPINGS = ('normal', 'exponential', 'gamma')
REFERENCE = PROJECT_DIR / 'results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42'
OUTPUT = PROJECT_DIR / 'results/11_nested_mapping_comparison/split_42_inner_1729/search_42'
LATENT_NAMES = ('u_f_low', 'u_f_high', 'u_window_size')


def latent_space():
    return [Real(0., 1., name=name) for name in LATENT_NAMES] + list(SEARCH_SPACE[3:])


def initial_coordinates(seed, count=16):
    # Use the identical row-wise random stream used for uniform's 50 candidates.
    unit = np.random.default_rng(seed).random((50, 5))
    physical = random_points(seed, 50)
    return [list(map(float, unit[i, :3])) + physical[i][3:] for i in range(count)]


def physical_point(point, mapping):
    preprocessing = transform_preprocessing_parameters(point[:3], mapping)
    return [preprocessing[name] for name in PARAMETERS[:3]] + list(map(float, point[3:]))


def validate_reference(reference, signals, labels):
    protocol = json.loads((reference/'protocol.json').read_text(encoding='utf-8'))
    status = json.loads((reference/'run_status.json').read_text(encoding='utf-8')) if (reference/'run_status.json').exists() else None
    # summary.json is portable and authoritative even if machine-local status was not synced.
    summary = json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    if summary['protocol'] != protocol:
        raise ValueError('Reference summary does not match its frozen protocol')
    if status is not None and status['status'] != 'complete':
        raise ValueError('Reference nested run is not complete')
    args = argparse.Namespace(outer_split_seed=protocol['outer_split_seed'],
                              inner_split_seed=protocol['inner_split_seed'], search_seed=protocol['search_seed'])
    if make_protocol(signals, labels, args) != protocol:
        raise ValueError('Current source, software or dataset differs from frozen uniform experiment')
    splits = make_splits(labels, protocol['outer_split_seed'], protocol['inner_split_seed'])
    if splits != json.loads((reference/'nested_folds.json').read_text(encoding='utf-8')):
        raise ValueError('Outer/inner folds differ from frozen uniform experiment')
    initial = random_points(protocol['search_seed'], 50)[:16]
    for split in splits:
        directory = reference/f"outer_fold_{split['outer_fold']}"
        history = pd.read_csv(directory/'bo_history.csv', float_precision='round_trip')
        if len(history) != 50:
            raise ValueError('Incomplete reference history')
        np.testing.assert_allclose(history[list(PARAMETERS)].iloc[:16].to_numpy(), initial, rtol=0, atol=1e-12)
        result = json.loads((directory/'bo_outer_result.json').read_text(encoding='utf-8'))
        predictions = result['recording_predictions']
        if sorted(p['recording_id'] for p in predictions) != sorted(split['test_segment_ids']):
            raise ValueError('Reference predictions do not match outer test IDs')
    return protocol, splits


def make_mapping_protocol(base, mapping):
    protocol = dict(base)
    protocol.update(protocol='nested_mapping_comparison_v1', parameter_mapping=mapping,
                    evaluations_per_method=50, initial_evaluations=16,
                    reference_uniform_protocol_sha256=_digest(base),
                    initialisation='Common random quantiles for preprocessing; identical C/gamma to uniform. '
                                   'Physical preprocessing configurations differ by mapping.',
                    scope='BO only; existing uniform results reused, no RS or fixed-reference reruns.')
    protocol.pop('shared_initial_evaluations', None)
    protocol['source_sha256'] = dict(base['source_sha256'])
    protocol['source_sha256']['nested_mapping_comparison.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['mapping_rules'] = {
        'normal': 'Truncated normal: midpoint location, range/4 scale',
        'exponential': 'Truncated exponential: lower-bound location, range/3 scale',
        'gamma': 'Truncated Gamma: shape=2, lower-bound location, range/4 scale',
    }
    protocol['latent_space'] = 'First three coordinates in [0,1]; inverse truncated CDF; '
    protocol['latent_space'] += 'C and SVM gamma remain log-uniform in original bounds.'
    protocol['cdf_endpoint_clipping'] = 1e-6
    return protocol


def save_state(directory, state, budget):
    temporary = directory/'search_state.joblib.tmp'
    joblib.dump(state, temporary, compress=0)
    temporary.replace(directory/'search_state.joblib')
    if state['history']:
        _csv_save(directory/'bo_history.csv', state['history'])
    _json_save(directory/'progress.json', dict(completed=len(state['history']), budget=budget,
                                              phase='search_complete' if len(state['history']) == budget else 'searching'))


def run_search(train_data, split, protocol, directory, max_new=None):
    """Only outer training data enter this function; outer data are not accepted."""
    signature = _digest(dict(protocol=protocol, split=split))
    checkpoint = directory/'search_state.joblib'
    state = joblib.load(checkpoint) if checkpoint.exists() else dict(
        signature=signature, history=[], optimizer=None)
    if state['signature'] != signature:
        raise ValueError('Checkpoint belongs to a different protocol')
    initial = initial_coordinates(protocol['search_seed'], protocol['initial_evaluations'])
    _immutable_json(directory/'initial_candidates.json', [dict(
        iteration=i+1, coordinates=point, parameters=dict(zip(PARAMETERS, physical_point(point, protocol['parameter_mapping']))))
        for i, point in enumerate(initial)])
    folds = [(np.asarray(f['train_segment_ids']), np.asarray(f['validation_segment_ids']))
             for f in split['inner_folds']]
    budget, n_initial = protocol['evaluations_per_method'], protocol['initial_evaluations']
    completed_now = 0
    while len(state['history']) < budget:
        if max_new is not None and completed_now >= max_new:
            return state, False
        iteration = len(state['history']) + 1
        started = perf_counter()
        if iteration <= n_initial:
            point = initial[iteration-1]
        else:
            if state['optimizer'] is None:
                optimizer = Optimizer(latent_space(), n_initial_points=n_initial,
                                      random_state=protocol['search_seed'], acq_func='EI',
                                      acq_func_kwargs={'xi': protocol['xi']}, acq_optimizer='lbfgs', model_queue_size=1)
                optimizer.tell(initial, [-row['score'] for row in state['history'][:n_initial]])
                state['optimizer'] = optimizer
            point = state['optimizer'].ask()
        optimizer_seconds = perf_counter()-started
        physical = physical_point(point, protocol['parameter_mapping'])
        row = _evaluate_candidate(train_data, folds, physical, iteration,
                                  'random_initial' if iteration <= n_initial else 'guided')
        row.update(dict(zip(LATENT_NAMES, map(float, point[:3]))))
        row['parameter_mapping'] = protocol['parameter_mapping']
        started = perf_counter()
        if iteration > n_initial:
            state['optimizer'].tell(point, -row['score'])
        row['optimizer_seconds'] = optimizer_seconds + perf_counter()-started
        state['history'].append(row)
        save_state(directory, state, budget)
        completed_now += 1
        print(f"{protocol['parameter_mapping']} fold {split['outer_fold']} BO {iteration}/{budget}: "
              f"CV={row['score']:.4f}, {row['candidate_seconds']:.1f}s", flush=True)
    save_state(directory, state, budget)
    return state, True


def summarize(output, reference):
    rows, predictions, histories = [], [], {}
    for mapping in ('uniform',)+MAPPINGS:
        for k in range(1, 6):
            directory = (reference if mapping == 'uniform' else output/mapping)/f'outer_fold_{k}'
            result_path = directory/'bo_outer_result.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding='utf-8'))
            selection = result['selection']
            metrics = result['segment_level_test_metrics']
            rows.append(dict(mapping=mapping, outer_fold=k, best_inner_cv_accuracy=selection['best_inner_cv_accuracy'],
                             best_iteration=selection['best_iteration'], outer_accuracy=metrics['accuracy'],
                             seizure_sensitivity=metrics['per_class_recall']['S'], **selection['parameters']))
            predictions.extend(dict(p, mapping=mapping, outer_fold=k) for p in result['recording_predictions'])
            history = pd.read_csv(directory/'bo_history.csv')
            histories[mapping, k] = np.maximum.accumulate(history.score.to_numpy())
    if rows:
        _csv_save(output/'mapping_results_per_fold.csv', rows)
        _csv_save(output/'mapping_outer_predictions.csv', predictions)
    if len(rows) != 20:
        return False
    table = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    summary = []
    for mapping in ('uniform',)+MAPPINGS:
        data = table[table.mapping == mapping]
        selected = pred[pred.mapping == mapping]
        assert sorted(selected.recording_id.tolist()) == list(range(500))
        summary.append(dict(mapping=mapping, folds=5, mean_outer_accuracy=float(data.outer_accuracy.mean()),
                            sd_outer_accuracy=float(data.outer_accuracy.std(ddof=1)),
                            mean_best_inner_cv_accuracy=float(data.best_inner_cv_accuracy.mean()),
                            sd_best_inner_cv_accuracy=float(data.best_inner_cv_accuracy.std(ddof=1)),
                            mean_best_at_16=float(np.mean([histories[mapping, k][15] for k in range(1,6)])),
                            mean_improvement_after_16=float(np.mean([histories[mapping, k][-1]-histories[mapping, k][15] for k in range(1,6)])),
                            pooled_seizure_sensitivity=float(data.seizure_sensitivity.mean())))
    _csv_save(output/'mapping_summary.csv', summary)
    pivot = table.pivot(index='outer_fold', columns='mapping', values='outer_accuracy')
    differences = [dict(outer_fold=int(k), mapping=mapping, difference_vs_uniform_pp=float(100*(row[mapping]-row['uniform'])))
                   for k, row in pivot.iterrows() for mapping in MAPPINGS]
    _csv_save(output/'paired_mapping_differences.csv', differences)
    _json_save(output/'summary.json', dict(methods=summary,
        note='Five dependent outer folds; one search seed per fold. Descriptive mapping comparison on previously inspected data.'))
    return True


def run(args):
    signals, labels = load_bonn_dataset(DATASET_PATH)
    base, splits = validate_reference(args.reference_run, signals, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output_dir):
        _immutable_json(args.output_dir/'nested_folds.json', splits)
        _immutable_json(args.output_dir/'reference_uniform_protocol.json', base)
        for mapping in MAPPINGS:
            directory = args.output_dir/mapping
            directory.mkdir(exist_ok=True)
            _immutable_json(directory/'protocol.json', make_mapping_protocol(base, mapping))
        if args.prepare_only:
            print('Verified frozen uniform source/data/software, all nested folds and three mapping protocols.', flush=True)
            return
        _json_save(args.output_dir/'run_status.json', dict(status='running', pid=os.getpid()))
        try:
            for mapping in MAPPINGS:
                protocol = make_mapping_protocol(base, mapping)
                for split in splits:
                    if args.folds and split['outer_fold'] not in args.folds:
                        continue
                    directory = args.output_dir/mapping/f"outer_fold_{split['outer_fold']}"
                    directory.mkdir(exist_ok=True)
                    _immutable_json(directory/'fold_manifest.json', split)
                    train_data = _select_segments(signals, labels, split['train_segment_ids'])
                    state, complete = run_search(train_data, split, protocol, directory, args.max_new_evaluations)
                    if not complete:
                        _json_save(args.output_dir/'run_status.json', dict(status='paused_at_evaluation_limit', pid=os.getpid()))
                        return
                    best = max(state['history'], key=lambda r: r['score'])
                    selection = dict(best_iteration=best['iteration'], best_inner_cv_accuracy=best['score'],
                                     parameters={name: best[name] for name in PARAMETERS})
                    _immutable_json(directory/'selected_configuration.json', selection)
                    signature = _digest(dict(protocol=protocol, split=split, selection=selection))
                    result_path = directory/'bo_outer_result.json'
                    if result_path.exists():
                        if json.loads(result_path.read_text())['signature'] != signature:
                            raise ValueError('Saved outer result belongs to different selection')
                    else:
                        test_data = _select_segments(signals, labels, split['test_segment_ids'])
                        result = evaluate_outer(train_data, test_data, selection['parameters'])
                        result.update(signature=signature, selection=selection, mapping=mapping,
                                      outer_fold=split['outer_fold'], completed_outer_evaluations=1)
                        _json_save(result_path, result)
                        print(f"{mapping} fold {split['outer_fold']}: outer predictions saved", flush=True)
                    summarize(args.output_dir, args.reference_run)
            complete = summarize(args.output_dir, args.reference_run)
            _json_save(args.output_dir/'run_status.json', dict(status='complete' if complete else 'partial_folds_complete', pid=os.getpid()))
            print('COMPLETE' if complete else 'PARTIAL', args.output_dir, flush=True)
        except BaseException as error:
            _json_save(args.output_dir/'run_status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                                             error=str(error), pid=os.getpid()))
            raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1,6))
    parser.add_argument('--max-new-evaluations', type=int)
    args = parser.parse_args()
    if args.max_new_evaluations is not None and args.max_new_evaluations < 1:
        parser.error('--max-new-evaluations must be positive')
    return args


if __name__ == '__main__':
    run(parse_args())
