"""Training-only, matched-budget screen of original_55 versus upstream catch24.

Uses only outer fold 1's 400 training recordings and its saved five inner folds.
No outer test signals or saved test predictions are read. This is exploratory
feature screening, not a nested outer-performance or BO-versus-RS experiment.
"""

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tmp/github_feature_deps'))
import joblib
import numpy as np
import pandas as pd
import pycatch22
from data_loader import _infer_class_from_path
from features import extract_features_by_setting
from grouped_bo_optimizer import _make_model, _majority_vote_by_segment, _segment_with_groups, _metrics
from preprocessing import bandpass_filter_signals

RUN = ROOT / 'results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42'
OUT = ROOT / 'results/12_training_feature_pilot/catch24_outer_fold_1'
FIXED = dict(f_low=0.5, f_high=40.0, window_size=2.0)
PARAMETERS = [dict(C=c, gamma=g) for c in (1., 10., 100.)
              for g in (0.0001, 0.001, 0.01, 0.1)]


def save_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_training():
    split = json.loads((RUN / 'nested_folds.json').read_text())[0]
    train_ids = np.array(split['train_segment_ids'])
    assert len(train_ids) == 400 and not set(train_ids) & set(split['test_segment_ids'])
    dataset = ROOT / 'Dataset'
    files = sorted(p for p in dataset.rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.txt', '.asc', '.csv'})
    files = [p for p in files if _infer_class_from_path(p, dataset) is not None]
    assert len(files) == 500
    # Enumerate file paths globally, but open only the 400 allowed training files.
    signals, labels, names = [], [], []
    for i in train_ids:
        path = files[i]
        signals.append(np.loadtxt(path, dtype=float).reshape(-1))
        labels.append(_infer_class_from_path(path, dataset))
        names.append(str(path.relative_to(dataset)))
    labels = np.array(labels)
    assert np.array_equal(np.bincount(labels), [80] * 5)
    seen = []
    for fold in split['inner_folds']:
        fit, validation = set(fold['train_segment_ids']), set(fold['validation_segment_ids'])
        assert len(fit) == 320 and len(validation) == 80
        assert not fit & validation and fit | validation == set(train_ids)
        seen.extend(validation)
    assert sorted(seen) == sorted(train_ids)
    return split, signals, labels, train_ids, names


def extract_catch24(windows):
    rows = []
    names = None
    for i, window in enumerate(windows):
        output = pycatch22.catch22_all(window, catch24=True)
        if names is None:
            names = output['names']
        assert names == output['names']
        rows.append(output['values'])
        if (i+1) % 500 == 0:
            print(f'catch24 features: {i+1}/{len(windows)} windows', flush=True)
    matrix = np.array(rows, dtype=float)
    assert matrix.shape == (len(windows), 24)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite catch24 features; investigate rather than silently replacing.')
    return matrix, names


def main(smoke=False):
    split, signals, labels, ids, files = load_training()
    started = perf_counter()
    filtered = bandpass_filter_signals(signals, 173.61, FIXED['f_low'], FIXED['f_high'])
    windows, y, groups = _segment_with_groups(filtered, labels, ids, FIXED['window_size'])
    preprocessing_seconds = perf_counter()-started
    if smoke:
        indices = [np.flatnonzero(y == label)[0] for label in range(5)]
        matrix, names = extract_catch24(windows[indices])
        original = extract_features_by_setting(windows[indices], 173.61, 'nf_enhanced')
        repeat, _ = extract_catch24(windows[indices])
        assert original.shape == (5,55) and np.isfinite(original).all()
        np.testing.assert_array_equal(matrix, repeat)
        print('Smoke passed: deterministic 24 features, baseline 55 features, 400 training IDs only.')
        print('Feature names:', names)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.sha256()
    for signal in signals:
        data_hash.update(signal.astype('<f8').tobytes())
    protocol = dict(
        experiment='exploratory_training_only_feature_screen_v1', outer_fold=1,
        source_repository='https://github.com/DynamicsAndNeuralSystems/pycatch22',
        upstream_definition='catch22_all(window, catch24=True); no manual feature selection',
        citation='Lubba et al. (2019), catch22: CAnonical Time-series CHaracteristics, doi:10.1007/s10618-019-00647-x',
        versions={p:version(p) for p in ['pycatch22','numpy','scipy','scikit-learn','PyWavelets']},
        fixed_preprocessing=FIXED, sampling_frequency=173.61,
        window_overlap=0., recording_vote='majority; ties choose smallest class index',
        candidates=PARAMETERS, budget_per_feature_set=12, search='fixed coarse grid; not BO',
        train_segment_ids=ids.tolist(), train_files=files,
        inner_folds=split['inner_folds'], training_data_sha256=data_hash.hexdigest(),
        preprocessing_seconds=preprocessing_seconds,
        outer_test_access=False,
        limitations='Only one outer training set, selected best CV is optimistic; not final performance.',
        sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                 ['pilot_github_features.py','features.py','preprocessing.py','data_loader.py','grouped_bo_optimizer.py']})
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Pilot output exists. Preserve it; use a new directory for another protocol.')
    save_json(OUT/'protocol.json', protocol)
    rows, feature_times = [], {}
    for setting in ['original_55', 'catch24']:
        save_json(OUT/'status.json', dict(stage='extracting', feature_setting=setting))
        started = perf_counter()
        if setting == 'original_55':
            X = extract_features_by_setting(windows, 173.61, 'nf_enhanced')
            names = [f'original_{i:02d}' for i in range(X.shape[1])]
            assert X.shape[1] == 55
        else:
            X, names = extract_catch24(windows)
        assert np.isfinite(X).all()
        feature_times[setting] = perf_counter()-started
        joblib.dump(dict(X=X,y=y,groups=groups,feature_names=names), OUT/f'{setting}_training_features.joblib',compress=3)
        print(f'{setting}: {X.shape}, extraction {feature_times[setting]:.2f}s', flush=True)
        for number, params in enumerate(PARAMETERS, 1):
            started = perf_counter()
            accuracies, recalls = [], []
            for inner in split['inner_folds']:
                fit = np.isin(groups, inner['train_segment_ids'])
                validation = np.isin(groups, inner['validation_segment_ids'])
                assert not np.any(fit & validation) and np.all(fit | validation)
                model = _make_model(params)
                model.fit(X[fit], y[fit])
                truth, predictions = _majority_vote_by_segment(model.predict(X[validation]), y[validation], groups[validation])
                assert len(truth) == 80
                result = _metrics(truth, predictions)
                accuracies.append(result['accuracy'])
                recalls.append(list(result['per_class_recall'].values()))
            row = dict(feature_setting=setting,candidate=number,**params,
                       mean_inner_cv_accuracy=float(np.mean(accuracies)),
                       fold_accuracy=accuracies,
                       candidate_seconds=perf_counter()-started)
            row.update({f'recall_{c}':float(v) for c,v in zip('ZONFS',np.mean(recalls,axis=0))})
            rows.append(row)
            save_json(OUT/'history.json',rows)
            pd.DataFrame(rows).to_csv(OUT/'history.csv',index=False)
            save_json(OUT/'status.json',dict(stage='searching',feature_setting=setting,candidate=number,total=12))
            print(f'{setting} {number}/12: C={params["C"]:g}, gamma={params["gamma"]:g}, CV={row["mean_inner_cv_accuracy"]:.4f}, {row["candidate_seconds"]:.1f}s',flush=True)
        del X
    summaries=[]
    for setting in feature_times:
        candidates=[r for r in rows if r['feature_setting']==setting]
        best=max(candidates,key=lambda r:r['mean_inner_cv_accuracy'])
        summaries.append(dict(feature_setting=setting,feature_columns=55 if setting=='original_55' else 24,
                              extraction_seconds=feature_times[setting],
                              search_seconds=sum(r['candidate_seconds'] for r in candidates),**{k:v for k,v in best.items() if k!='feature_setting'}))
    save_json(OUT/'summary.json',dict(results=summaries,outer_test_access=False))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    save_json(OUT/'status.json',dict(stage='complete',outer_test_access=False))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
