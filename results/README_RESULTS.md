# Results Folder Index

## Current: 09_fixed_fold_segment_accuracy

This is the current recording-fixed grouped-CV protocol with
`segment_accuracy` as the BO objective. Each completed run must contain
`grouped_cv_folds.json`, `grouped_bo_history.csv`, and
`grouped_bo_final_result.json` with `cv_split_protocol=recording_fixed_v2`.
The five-seed, four-prior experiment is complete. `repeated/` contains the
per-seed values, means, SDs, descriptive intervals, and paired differences;
`method_comparison_seed_42/` contains the fixed-reference, random-search
and BO comparison. The seed 43–46 comparison folders contain the matching
random-search histories, final results and verified split and fold manifests.
`paired_bo_vs_random_seeds_42_46.csv` summarises all five BO/RS pairs.
`visualizations/seed_42_uniform/` contains current surrogate plots. Do not
combine these results with the archived results below in one performance
comparison.

All folders listed below are historical and are not current thesis evidence.
In particular, `00_main_bo_svm` used a different validation protocol, while
`06_grouped_prior_comparison`, `07_grouped_segment_objective`, and
`08_surrogate_visualization` predate the recording-fold fix.

This folder is organized by experiment branch.

## 00_main_bo_svm

Archived direct BO-SVM results:

- `baseline_metrics.csv`
- `bo_history.csv`
- `bo_best_params.csv`
- `random_search_history.csv`
- `random_search_best_params.csv`
- `comparison_summary.csv`
- `bo_best_confusion_matrix.png`
- `convergence_bo_vs_random.png`
- `test_accuracy_comparison.png`
- `class_wise_recall_comparison.png`

Historical PPT comparison only:

```text
SVM without BO
SVM with BO
Random Search SVM
```

## 01_subset_tasks

Subset classification experiments using full BO:

- `subset_bo_results.csv`
- `subset_bo_history.csv`
- `subset_bo_table3_metrics.csv`
- subset accuracy, convergence, and confusion-matrix plots

Use this folder for:

```text
A vs E / Z vs S
D vs E / F vs S
A vs D vs E / Z vs F vs S
```

## 02_extended_svm_archive

Old result files kept as archive only. These are not the current main thesis
line.

## 80_legacy_outputs

Older output folders and notebook checkpoint artifacts.

## 90_bo_archives

Old BO running histories and backup histories.

## Note

The archived scripts wrote directly into their matching result folders:

- main BO-SVM scripts -> `results/00_main_bo_svm`
- subset full BO scripts -> `results/01_subset_tasks`
