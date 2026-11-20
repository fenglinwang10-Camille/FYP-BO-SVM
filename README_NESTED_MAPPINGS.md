# Nested BO parameter-mapping comparison

This completes the mapping comparison on the exact five outer folds and inner
folds used by the completed uniform BO experiment. It adds normal, exponential
and Gamma BO: five searches per mapping, 50 candidates per search, 16 random
initial candidates and 34 EI-guided candidates. There are 750 new candidate
evaluations. Existing uniform, RS and fixed-reference results are not rerun.

The bounds, 55 features, filtering/windowing, scaler/SVM, majority-vote tie rule,
EI xi=0.01, search seed 42, and GP optimiser settings remain unchanged. The new
runner checks all original source hashes, dependency versions, the dataset hash,
and every saved outer/inner partition against the completed uniform protocol.
The original frozen files are imported but not modified.

The first three BO coordinates are quantiles in [0,1], transformed using the
existing bounded inverse-CDF functions in parameter_priors.py. These use the
previously specified parameters: normal midpoint and range/4 scale;
exponential lower-bound location and range/3 scale; Gamma shape 2,
lower-bound location and range/4 scale. Endpoints are clipped at 1e-6. The GP
operates on these mapped coordinates. C and SVM gamma remain log-uniform.

All mappings use the same initial random quantiles and exactly the same initial
C/gamma values as uniform. Initial physical preprocessing settings differ by
mapping, as intended. Forcing identical physical settings across mappings would
remove the intended difference in initial sampling. Both sampling and coordinate
geometry can influence the results; this is not a pure initial-sampling ablation.

## Validate and launch

```powershell
python -B -m unittest test_nested_mapping_comparison
python -B nested_mapping_comparison.py --prepare-only
powershell -ExecutionPolicy Bypass -File .\run_nested_mappings.ps1
```

The launcher uses the existing eegbo interpreter when available. It runs one
worker with one BLAS thread and starts a hidden background process. Keep the
computer powered and awake. The chat does not need to remain open. Re-run the
launcher to resume after interruption; an OS lock prevents concurrent writes.

Results go to:

```text
results/11_nested_mapping_comparison/split_42_inner_1729/search_42/
```

Each mapping has an immutable protocol.json, then outer_fold_1 through
outer_fold_5 containing the fold IDs, initial candidates, history, atomic
checkpoint, progress, frozen selected configuration and final outer predictions.
Check run_status.json and the latest run_*.log for progress. Search calls accept
only outer training data. Outer test prediction happens after the full budget
and saved selection. Completed outer results are reused, not rescored.

Once all 15 searches finish, mapping_summary.csv, paired_mapping_differences.csv,
mapping_results_per_fold.csv, mapping_outer_predictions.csv and summary.json
combine the three new mappings and the saved uniform results. Intermediate
per-fold tables are partial; only summary.json signals a complete comparison.

One search seed per fold remains a limitation. Outer test partitions are
disjoint but outer training sets overlap. Subject IDs are unavailable. These are
additional analyses of already inspected data, not new external validation.
Do not alter mappings or bounds based on outer scores while the run progresses.

For a resumable single-candidate training check:

```powershell
python -B -u nested_mapping_comparison.py --folds 1 --max-new-evaluations 1
```

This stops before any outer-test prediction and can be resumed by the launcher.
