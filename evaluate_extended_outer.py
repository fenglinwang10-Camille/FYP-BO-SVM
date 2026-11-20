"""Evaluate frozen 100-candidate selections without further parameter search."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nested_bo_rs as base
import nested_score_tiebreak as voting

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/14_training_search_extension_100/split_42_inner_1729/search_42'
EARLIER = ROOT / 'results/13_nested_score_tiebreak/split_42_inner_1729/search_42'
OUTPUT = ROOT / 'results/16_extended_100_outer_evaluation/split_42_inner_1729/search_42'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    original = read(SOURCE / 'protocol.json')
    assert read(SOURCE / 'run_status.json')['status'] == 'complete'
    for name, digest in original['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name, expected in original['versions'].items():
        assert version(name) == expected, name
    signals, labels = base.load_bonn_dataset(base.DATASET_PATH)
    data_hash = hashlib.sha256()
    for signal, label in zip(signals, labels):
        values = np.asarray(signal, dtype='<f8')
        data_hash.update(np.asarray([len(values), int(label)], dtype='<i8').tobytes())
        data_hash.update(values.tobytes())
    assert data_hash.hexdigest() == original['dataset_sha256']
    splits = read(SOURCE / 'nested_folds.json')
    assert splits == base.make_splits(labels, original['outer_split_seed'], original['inner_split_seed'])
    selections = {}
    for split in splits:
        fold = split['outer_fold']
        directory = SOURCE / f'outer_fold_{fold}'
        selected = read(directory / 'selected_training_configurations.json')
        for method in ('bo', 'rs'):
            history = pd.read_csv(directory / f'{method}_history.csv')
            assert len(history) == 100
            best = history.iloc[int(np.argmax(history.score.to_numpy()))]
            assert int(best.iteration) == selected[method]['best_iteration']
            assert abs(float(best.score) - selected[method]['best_inner_cv_accuracy']) < 1e-12
            np.testing.assert_allclose([best[p] for p in base.PARAMETERS],
                                       [selected[method]['parameters'][p] for p in base.PARAMETERS],
                                       rtol=1e-12, atol=1e-12)
        selected['fixed_reference'] = dict(parameters=original['fixed_reference_parameters'],
                                           best_iteration=None, best_inner_cv_accuracy=None)
        selections[str(fold)] = selected
    protocol = copy.deepcopy(original)
    protocol.update(protocol='frozen_extended_100_outer_evaluation_v1', outer_test_access=True,
                    purpose='Exploratory outer evaluation of frozen 100-candidate selections; not new external validation',
                    selection_source=str(SOURCE.relative_to(ROOT)),
                    source_protocol_sha256=hashlib.sha256((SOURCE / 'protocol.json').read_bytes()).hexdigest(),
                    reuse_policy='Identical selected parameters reuse verified experiment-13 outer predictions; others refitted once')
    protocol['source_sha256']['evaluate_extended_outer.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with base.run_lock(OUTPUT):
        base._immutable_json(OUTPUT / 'protocol.json', protocol)
        base._immutable_json(OUTPUT / 'nested_folds.json', splits)
        # Freeze all five folds' selections before any outer model evaluation.
        base._immutable_json(OUTPUT / 'all_selected_configurations.json', selections)
        base._json_save(OUTPUT / 'run_status.json', dict(status='running', pid=os.getpid()))
        try:
            for split in splits:
                fold = split['outer_fold']
                directory = OUTPUT / f'outer_fold_{fold}'
                directory.mkdir(exist_ok=True)
                selected = selections[str(fold)]
                base._immutable_json(directory / 'selected_configurations.json', selected)
                for method, selection in selected.items():
                    signature = base._digest(dict(protocol=protocol, split=split, selection=selection))
                    path = directory / f'{method}_outer_result.json'
                    if path.exists():
                        assert read(path)['signature'] == signature
                        continue
                    prior_path = EARLIER / directory.name / path.name
                    prior = read(prior_path)
                    expected_signature = base._digest(dict(protocol=read(EARLIER / 'protocol.json'),
                                                           split=split, selection=prior['selection']))
                    assert prior['signature'] == expected_signature
                    if prior['selection']['parameters'] == selection['parameters']:
                        result = copy.deepcopy(prior)
                        result['prediction_provenance'] = dict(reused_from=str(prior_path.relative_to(ROOT)),
                                                               source_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest())
                    else:
                        result = voting.evaluate_outer(
                            base._select_segments(signals, labels, split['train_segment_ids']),
                            base._select_segments(signals, labels, split['test_segment_ids']), selection['parameters'])
                        result['prediction_provenance'] = dict(refitted_for_budget=100)
                    assert sorted(p['recording_id'] for p in result['recording_predictions']) == sorted(split['test_segment_ids'])
                    result.update(signature=signature, method=method, outer_fold=fold, selection=selection,
                                  completed_outer_evaluations=1)
                    base._json_save(path, result)
                    print(f"Fold {fold} {method}: accuracy={result['segment_level_test_metrics']['accuracy']:.4f}", flush=True)
                base.summarize(OUTPUT, protocol)
            assert base.summarize(OUTPUT, protocol)
            base._json_save(OUTPUT / 'run_status.json', dict(status='complete', pid=os.getpid()))
        except BaseException as error:
            base._json_save(OUTPUT / 'run_status.json', dict(status='failed', error=str(error), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
