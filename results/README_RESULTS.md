# Retained experiment results

## 13_nested_score_tiebreak

Completed supplementary nested BO/RS/reference comparison using the same
splits, bounds and budgets as experiment 10, but resolving recording majority
vote ties by mean SVM OVR scores among tied classes. The rule was selected
after a training-only pilot, so these are exploratory results. Five-fold outer
accuracy: BO 88.0% (SD 2.83), RS 87.4% (SD 5.22), fixed reference 86.6%
(SD 7.50). Original experiment 10 remains unchanged. See
`../README_NESTED_SCORE_TIEBREAK.md`.

## 12_training_feature_pilot

Training-only screens on the 400 outer-fold-1 training recordings and the
same five inner folds, with twelve matched SVM settings. Contains original55
versus catch24, basic23, and three recording-vote rules. No outer evaluation.
CSV/JSON records are retained; regenerable feature matrices are excluded.
See `../README_GITHUB_FEATURE_PILOT.md`.

## 11_nested_mapping_comparison

Adds normal, exponential and Gamma BO on the same frozen outer/inner folds as
10_nested_cv_shared_initialization. Five 50-candidate searches per new mapping,
16 initial candidates and 34 EI-guided candidates. Initial preprocessing
quantiles and C/gamma draws are matched; physical preprocessing values differ
by the bounded inverse-CDF mapping. Existing uniform results are reused.
See ../README_NESTED_MAPPINGS.md. Only the final summary.json represents the
complete four-mapping comparison; intermediate per-fold tables may be partial.

## 10_nested_cv_shared_initialization

Outer five-fold evaluation with fixed inner five-fold CV. BO and RS share 16
initial candidates and their scores, then evaluate 34 further candidates each.
Outer/inner split seeds and search seeds are separate. Existing preprocessing,
55 features and majority-vote rules are retained. See
`../README_NESTED_EXPERIMENT.md`. The local `run_status.json` records progress;
`summary.json` is created after all five outer folds finish. This is a structured
reanalysis of previously inspected recordings, not external validation.

## 09_fixed_fold_segment_accuracy

Exploratory repeated holdouts under `recording_fixed_v2`. The retained results
include seeds 42-46 for all four BO mappings, paired RS comparisons, the fixed
reference, repeated-run summaries, search-behaviour analyses, GP diagnostics,
parameter/surrogate visualizations, the training-only voting tie-break analysis,
and the window-length training-CV check. Each original run used 50 evaluations
and 16 initial points. These holdout test sets overlap and were inspected during
development. They must not be presented as independent external datasets or used
to choose new mappings and configurations.

## raw_psd_inspection

Unfiltered signal-spectrum diagnostics used to check the preprocessing context.
Training-only PSD outputs are also retained in `../chapter4_draft/training_psd/`.

Historical protocols (00-08, 80 and 90), subset tasks and unused supplementary
feature/aggregation/soft-voting experiments have been removed from this checkout
and the repository's current version. Older commits retain their history.
