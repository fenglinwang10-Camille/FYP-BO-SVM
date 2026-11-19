# Nested five-fold BO/RS comparison

This experiment implements an outer five-fold evaluation and an inner five-fold
parameter search. It uses the existing 55 features, preprocessing bounds,
log-uniform C and gamma, and majority voting with the original smallest-index
tie rule. No feature selection or alternative voting rule is introduced here.

The outer split is made on the 500 original recording IDs. Each outer test fold
contains 100 recordings, 20 from each class. The other 400 recordings are split
into five fixed inner folds (320 training and 80 validation recordings). Windows
from an original recording always remain together. The scaler is fitted inside
each training fold. Recording IDs do not identify subjects.

BO and RS receive the same first 16 configurations and exactly the same inner
CV scores, computed once and reused. BO uses GP/EI with xi=0.01 for its remaining
34 configurations; RS continues independent sampling for its remaining 34.
Each method has 50 candidates, requiring 84 distinct training-CV evaluations
per outer fold, or 420 across the five folds. The fixed reference receives no
search and is fitted on the same outer training recordings.

Both searches finish before their configurations are selected. Ties select the
earliest highest-scoring candidate. The selections are saved before any outer
test features are extracted. Each selected model is fitted on the 400 outer
training recordings and predicts the 100 outer test recordings. Every recording
has exactly one outer prediction per method after all five folds finish.

## Start or resume in the background

From the project folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nested_experiment.ps1
```

The launcher uses D:\Anaconda3\envs\eegbo\python.exe if available, otherwise
Python on PATH. On another device, install requirements.txt first. An explicit
interpreter can be supplied with `-PythonPath 'C:\path\to\python.exe'`.
The process runs in a hidden window, with a single worker and one BLAS thread.
Keep the computer powered and awake; this launcher does not change sleep settings.

The default seeds are outer split 42, inner split 1729 (plus outer-fold index
minus one), and search 42. They are recorded separately. To add an optimizer
replicate on the same folds, use `-SearchSeed 43`. This creates a separate output
directory; it does not create an independent dataset.

The default output is:

```text
results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42/
```

`protocol.json` freezes parameters, software versions, source hashes and the data
hash. `nested_folds.json` saves all outer and inner IDs. Each `outer_fold_*`
directory contains the shared initial candidates, shared/BO/RS histories,
progress.json, selected configurations, and the three completed outer results.
The outer results include the recording predictions and confusion matrices.

The process saves an atomic checkpoint after every completed candidate. Run the
same command to resume. Finished outer results are reused without rescoring.
An interruption before an atomic save can require repeating the unfinished
calculation. A changed protocol, source or dataset is rejected rather than mixed
with saved results. The OS lock prevents two processes writing the same run.

`run_status.json` reports running, failed, interrupted, or complete. Logs and the
background PID are saved in the run directory. When complete, the experiment
automatically writes `outer_fold_results.csv`, `paired_bo_minus_rs.csv`,
`outer_recording_predictions.csv`, and `summary.json`. Partial fold tables may be
written earlier; only `summary.json` represents the complete five-fold result.
The SD describes variation across folds; it is not evidence of five independent
datasets. Accuracy values in saved tables are proportions (0 to 1).

## Validate without a full search

```powershell
python -B -m unittest test_nested_bo_rs test_grouped_segment_objective
python -B nested_bo_rs.py --prepare-only
```

To evaluate just the first shared candidate of fold 1 and save it for resumption:

```powershell
python -B -u nested_bo_rs.py --folds 1 --max-new-evaluations 1
```

This limit never finalizes a shortened search or evaluates outer test data.
Resume without the limit to finish the full protocol. Trusted local checkpoint
files contain Python objects and must not be loaded from an untrusted source.

## Interpretation

The old 09_fixed_fold_segment_accuracy results remain exploratory historical
results. This new protocol was fixed after inspecting the same dataset in prior
development, so it is a structured reanalysis rather than an untouched external
validation. It does not establish performance on unseen subjects. Outer test
scores must not be used to change mappings, bounds or reporting choices during
the run. There is one paired optimizer run per outer fold by default.
