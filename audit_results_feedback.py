"""Post-hoc reproducibility and paired-prediction audit; no search is rerun.

The frozen nested-experiment source files and results are not modified.
--reproduce-seed44 additionally recomputes the historical fixed reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42'


def paired_counts(bo_correct, rs_correct):
    bo, rs = np.asarray(bo_correct, dtype=bool), np.asarray(rs_correct, dtype=bool)
    if bo.shape != rs.shape or bo.ndim != 1 or not len(bo):
        raise ValueError('Expected equal, nonempty paired correctness vectors')
    b, c = int((bo & ~rs).sum()), int((~bo & rs).sum())
    return dict(both_correct=int((bo & rs).sum()), bo_only_correct=b,
                rs_only_correct=c, both_wrong=int((~bo & ~rs).sum()),
                accuracy_difference_pp=float(100 * (bo.mean() - rs.mean())),
                mcnemar_exact_two_sided_p=float(binomtest(b, b+c, .5).pvalue) if b+c else 1.)


def analyse_nested():
    data = pd.read_csv(RESULTS / 'outer_recording_predictions.csv')
    if data.duplicated(['recording_id', 'method']).any():
        raise ValueError('Duplicated out-of-fold predictions')
    if data.groupby('recording_id').true_label.nunique().max() != 1:
        raise ValueError('Inconsistent truth labels')
    if data.groupby('recording_id').outer_fold.nunique().max() != 1:
        raise ValueError('Methods were not evaluated on the same outer fold')
    data['correct'] = data.true_label == data.predicted_label
    wide = data.pivot(index='recording_id', columns='method', values='correct')
    assert wide.index.tolist() == list(range(500)) and not wide.isna().any().any()
    folds = []
    for k in range(1, 6):
        directory = RESULTS / f'outer_fold_{k}'
        selected = json.loads((directory / 'selected_configurations.json').read_text())
        same = selected['bo']['parameters'] == selected['rs']['parameters']
        subset = data[data.outer_fold == k]
        current = subset.pivot(index='recording_id', columns='method', values='correct')
        assert len(current) == 100
        entry = dict(outer_fold=k, same_configuration=same,
                     bo_best_iteration=selected['bo']['best_iteration'],
                     rs_best_iteration=selected['rs']['best_iteration'],
                     **paired_counts(current.bo, current.rs))
        if same:
            predictions = subset.pivot(index='recording_id', columns='method', values='predicted_label')
            assert (predictions.bo == predictions.rs).all()
        folds.append(entry)
    report = dict(analysis='Post-hoc paired out-of-fold prediction analysis',
                  n_recordings=500, pooled=paired_counts(wide.bo, wide.rs), folds=folds,
                  identical_configuration_folds=[r['outer_fold'] for r in folds if r['same_configuration']],
                  caveat='The McNemar p-value is nominal and assumes independent recording pairs. '
                         'Overlapping outer training sets and unavailable subject IDs limit this assumption. '
                         'It is not a definitive test of superiority across subjects or datasets.',
                  prediction_sha256=hashlib.sha256((RESULTS/'outer_recording_predictions.csv').read_bytes()).hexdigest())
    destination = RESULTS / 'posthoc_audit'
    destination.mkdir(exist_ok=True)
    (destination/'paired_prediction_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(folds).to_csv(destination/'paired_fold_details.csv', index=False)
    return report


def reproduce_seed44():
    from grouped_bo_optimizer import _cross_validate_training_segments, _final_test_once
    from grouped_method_comparison import MANUAL_PARAMS, _load_shared_data, _result_payload
    audit_dir = ROOT / 'results/09_fixed_fold_segment_accuracy/reproducibility_audit_seed44'
    audit_dir.mkdir(exist_ok=True)
    training, testing = _load_shared_data(44, audit_dir)
    score, validation, _, _ = _cross_validate_training_segments(
        *training, MANUAL_PARAMS, objective_mode='segment_accuracy', random_state=44)
    window, segment, _, _ = _final_test_once(*training, *testing, MANUAL_PARAMS)
    result = _result_payload('Manual SVM', MANUAL_PARAMS, score, validation, window, segment)
    result.update(random_state=44, objective_mode='segment_accuracy', cv_split_protocol='recording_fixed_v2')
    old_path = audit_dir.parent/'method_comparison_seed_44/grouped_manual_final_result.json'
    original = json.loads(old_path.read_text())

    def check(actual, expected, path='result'):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys(), path
            for key in expected: check(actual[key], expected[key], path+'.'+key)
        elif isinstance(expected, (int, float, list)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=path)
        else:
            assert actual == expected, path
    check(result, original)
    (audit_dir/'reproduced_manual_final_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = dict(seed=44, all_saved_fields_match=True, absolute_tolerance=1e-12,
                  validation_segment_accuracy=score, test_segment_accuracy=segment['accuracy'],
                  script='python -B audit_results_feedback.py --reproduce-seed44',
                  original_json_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Recomputed for reproducibility only; original JSON was not overwritten. '
                       'This verifies its numerical contents, not the provenance of the original generation.')
    (audit_dir/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduce-seed44', action='store_true')
    args = parser.parse_args()
    print(json.dumps(analyse_nested(), indent=2))
    if args.reproduce_seed44:
        print(json.dumps(reproduce_seed44(), indent=2))
