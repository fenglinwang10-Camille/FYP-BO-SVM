"""Recreate the regenerable 55-D pilot cache from saved training files only."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from data_loader import _infer_class_from_path
from features import extract_features_by_setting
from grouped_bo_optimizer import _segment_with_groups
from preprocessing import bandpass_filter_signals


def main():
    root=Path(__file__).resolve().parent
    directory=root/'results/12_training_feature_pilot/catch24_outer_fold_1'
    protocol=json.loads((directory/'protocol.json').read_text())
    assert protocol['outer_test_access'] is False
    for name in ('features.py','preprocessing.py','data_loader.py','grouped_bo_optimizer.py'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==protocol['sources'][name]
    target=directory/'original_55_training_features.joblib'
    if target.exists():
        data=joblib.load(target)
        assert data['X'].shape==(4400,55) and np.isfinite(data['X']).all()
        assert set(data['groups'])==set(protocol['train_segment_ids'])
        print('Existing training cache verified; no overwrite needed.')
        return
    paths=[root/'Dataset'/name for name in protocol['train_files']]
    assert len(paths)==400
    signals=[np.loadtxt(path).reshape(-1) for path in paths]
    digest=hashlib.sha256()
    for signal in signals:
        digest.update(signal.astype('<f8').tobytes())
    assert digest.hexdigest()==protocol['training_data_sha256']
    labels=np.array([_infer_class_from_path(path,root/'Dataset') for path in paths])
    ids=np.array(protocol['train_segment_ids'])
    prep=protocol['fixed_preprocessing']
    filtered=bandpass_filter_signals(signals,protocol['sampling_frequency'],prep['f_low'],prep['f_high'])
    windows,y,groups=_segment_with_groups(filtered,labels,ids,prep['window_size'])
    X=extract_features_by_setting(windows,protocol['sampling_frequency'],'nf_enhanced')
    assert X.shape==(4400,55) and np.isfinite(X).all()
    joblib.dump(dict(X=X,y=y,groups=groups,feature_names=[f'original_{i:02d}' for i in range(55)]),target,compress=3)
    print('Rebuilt training-only cache:',target)


if __name__=='__main__':
    main()
