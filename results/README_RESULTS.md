# Retained experiment results

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
