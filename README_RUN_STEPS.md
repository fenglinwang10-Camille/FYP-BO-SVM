# Current Bonn EEG BO-SVM Pipeline

The new nested five-fold experiment with shared BO/RS initialization is described
in `README_NESTED_EXPERIMENT.md`. Start or resume it in the background with
`powershell -ExecutionPolicy Bypass -File .\run_nested_experiment.ps1`.
It writes to `results/10_nested_cv_shared_initialization/`. The commands below
document the earlier repeated-holdout experiment; their results remain intact.

The current experiment uses the five-class Bonn EEG recordings, an 80/20
stratified outer split by original recording, five CV folds fixed at the
recording level, 55 `nf_enhanced` features, and an RBF-SVM. Bayesian
optimisation maximizes **recording-level (segment) validation accuracy**
after majority voting over each recording's windows. The held-out test set
is evaluated once per completed search. The test set must not be used to
select the reported prior or tune the search design.

Install: `pip install numpy scipy scikit-learn scikit-optimize pandas matplotlib PyWavelets`

Run one seed for all four preprocessing search distributions:

```bash
python run_grouped_segment_priors.py --seed 42
```

Repeat with seeds 43, 44, 45, and 46, then summarize:

```bash
python summarize_grouped_priors.py --seeds 42 43 44 45 46
python plot_grouped_method_comparison.py
python plot_grouped_surrogate.py --seed 42 --prior uniform
```

These commands require the dataset in `Dataset/`. BO runs use 50 candidate
evaluations and 16 initial points each. All current results go under
`results/09_fixed_fold_segment_accuracy/`, with a recording-fold manifest and
`cv_split_protocol=recording_fixed_v2`. `grouped_method_comparison.py` defaults
to seed 42 and includes the fixed reference. To reproduce the additional
BO versus random-search comparisons, run:

```powershell
foreach ($seed in 43..46) { python grouped_method_comparison.py --seed $seed }
```

Each comparison requires the completed uniform BO result for that seed.
The runner checks both the outer split and the five recording folds against
the saved BO manifests. It reuses the BO result and selects the random search
configuration by training CV accuracy from 50 evaluations. Additional seeds
omit the fixed reference. Each seed writes to `method_comparison_seed_<seed>/`.
The completed seeds 42–46 are summarised in
`results/09_fixed_fold_segment_accuracy/paired_bo_vs_random_seeds_42_46.csv`,
including CV accuracy, test accuracy, seizure sensitivity counts and N/F errors.

As of 2026-10-20, all 20 BO runs (seeds 42-46, four priors), the five-seed
summary, the seed-42 method comparison, and the seed-42 uniform surrogate
plots are complete. Mean held-out recording accuracy over five splits is
86.8% (uniform), 84.8% (normal), 84.4% (exponential), and 85.0% (gamma).
The uniform mean is 1.8–2.4 percentage points above the other observed
means. Sample SDs range from 2.0 to 3.6 percentage points, and each seed
changes both the split and search trajectory. These results describe five
overlapping outer splits. The outer test results were inspected during
development. Use training CV or separate validation data to select a mapping
or configuration in future experiments.

### Training-CV feature screen

```bash
python analyze_grouped_features.py
```

This fixed-configuration, training-only screen compares the four existing
feature sets using the completed recording-fold manifests. Its results are
under `results/09_fixed_fold_segment_accuracy/feature_ablation_fixed_config/`.
The reference parameters are `f_low=1`, `f_high=40`, `window_size=5`, `C=10`,
and `gamma=0.01`; no feature setting receives a new BO search, and the
held-out test set is not scored. Over five outer training splits, mean CV
recording accuracy is 80.50% (`original`), 82.85% (`entropy_hjorth`), 83.05%
(`expanded`), and 85.45% (`nf_enhanced`). For `nf_enhanced`, mean F recall is
54.25%; 168 of 400 F recording instances across the overlapping splits were
classified as N. These CV numbers must not be compared directly with the
BO-selected held-out test accuracies above.

### Exploratory vote tie-break screen

```bash
python analyze_grouped_voting.py
```

This reproduces each seed's selected uniform-BO training-CV score, then
compares the existing smallest-class-index tie rule with a rule that uses
mean SVM decision scores only among tied classes. Over five overlapping
training splits, mean CV recording accuracy changed from 88.90% to 89.25%
and mean F recall from 69.75% to 72.25%. One seed worsened. No held-out
test recording was scored, and this small exploratory gain is not a new
validated test result. Outputs are in
`results/09_fixed_fold_segment_accuracy/voting_tie_break_training_cv/`.

The older `results/00_*`, `04_*`, `06_*`, `07_*`, and `08_*` outputs and the
commands below are **archived**. In particular, previously quoted accuracy
figures and surrogate plots were produced before the fold fix and cannot be
presented as results of the current protocol. The MIT dataset is not
part of the current validation plan.

## Archived Instructions (Do Not Use For Current Results)

# Bonn EEG Direct BO-SVM Pipeline

This project compares:

- SVM without Bayesian Optimization
- SVM with Bayesian Optimization
- SVM with Random Search

The current thesis line is direct BO-SVM. The model is still an RBF-SVM; BO is
used to tune signal preprocessing and SVM hyperparameters.

## Install

```bash
pip install numpy scipy scikit-learn scikit-optimize pandas matplotlib PyWavelets
```

`PyWavelets` is optional. The current fixed feature setting does not require
wavelet coefficient features.

## Dataset

Place the Bonn EEG dataset inside:

```text
Dataset
```

The five classes are:

```text
Z = 0
O = 1
N = 2
F = 3
S = 4
```

## Run Order

```bash
python baseline_manual.py
python bo_optimizer.py
python random_search.py
python bo_best_evaluation.py
python compare_results.py
python plot_results.py
python subset_bo_experiments.py
python make_subset_table3_metrics.py
python plot_subset_results.py
python meeting_results_summary.py
```

### Compare preprocessing parameter priors

The default command keeps the original uniform preprocessing search:

```bash
python bo_optimizer.py --parameter-prior uniform
```

The additional experiments use an inverse-CDF transformation for `f_low`,
`f_high`, and `window_size`:

```bash
python bo_optimizer.py --parameter-prior normal
python bo_optimizer.py --parameter-prior exponential
python bo_optimizer.py --parameter-prior gamma
```

All distributions are truncated to the original parameter bounds. Normal uses
the range midpoint and `range / 4` standard deviation; Exponential uses the
lower bound as its location and `range / 3` scale; Gamma uses shape `2`, the
lower bound as its location, and `range / 4` scale. `C` and SVM `gamma` remain
log-uniform. Non-uniform results are saved under
`results/02_prior_experiments/<prior>` so the established main result is not
overwritten.

For a controlled comparison using the original result's `nf_enhanced` feature
setting, run `run_prior_experiments.ps1`. It runs all four distributions with
the same 80-call budget, 16 initial points, random seed 42, five-fold CV,
and objective score. Results go to `results/03_prior_comparison_nf_enhanced`.
The older main BO history mixes feature settings, so its best row is not a
valid uniform baseline for this controlled comparison.

### Compare grouped preprocessing priors

Keep the grouped N/F-focused uniform run in `results/04_grouped_bo_svm` as the
seed-42 baseline. The commands below change only the bounded search
distribution of `f_low`, `f_high`, and `window_size`. They retain the same
80/20 recording-level split, 5 grouped CV folds, 50 BO evaluations, 16
initial evaluations, feature setting, class weighting, objective, and SVM
parameter bounds. SVM `gamma` is a separate hyperparameter and stays
log-uniform. Non-uniform runs are saved separately and can resume after a stop.

```bash
python grouped_bo_optimizer.py --parameter-prior normal
python grouped_bo_optimizer.py --parameter-prior exponential
python grouped_bo_optimizer.py --parameter-prior gamma
python compare_grouped_priors.py
```

The inverse-CDF transform also changes the GP's coordinate geometry; this is
a comparison of BO search parameterizations, not a pure change to a Bayesian
prior over model parameters. Incomplete runs are displayed without test scores.

### Compare grouped BO objectives

The grouped N/F-weighted BO run is saved in `results/04_grouped_bo_svm`.
Run the matched macro-F1 objective with the same split, search space, 50
evaluations, 16 initial points, and random seed:

```bash
python grouped_bo_optimizer.py --objective-mode macro_f1
python compare_grouped_objectives.py
```

The macro-F1 run is saved in `results/05_grouped_objective_comparison/macro_f1`.
The comparison CSV checks the outer split, common settings, complete histories,
and identical initial candidates before reporting held-out window and segment
metrics. This is one outer split; repeat outer splits before interpreting small
differences as stable improvements.

### Compare segment-level accuracy objective

The original N/F-weighted objective scores individual validation windows,
although the primary reported result is accuracy per original EEG recording.
This experiment keeps the original seed, search space, features, grouped folds,
and 50-call budget; only the BO score changes. Within each validation fold,
window predictions are combined by the same majority vote used on the test
recordings, then accuracy is calculated over recordings.

```bash
python grouped_bo_optimizer.py --objective-mode segment_accuracy
python compare_grouped_segment_objective.py
```

For an additional paired outer split, pass the same seed to both commands.
The comparison validates the outer split and first 16 BO candidates. Results
are stored separately under `results/07_grouped_segment_objective`.

To compare all four preprocessing search distributions with this newer
segment-accuracy objective, run:

```bash
python run_grouped_segment_priors.py --seed 42
```

The completed uniform run is reused. Missing normal, exponential, and gamma
runs are saved under `results/06_grouped_prior_comparison/seed_42/segment_accuracy`.
The runner verifies completed runs, resumes partial histories, and writes
`results/06_grouped_prior_comparison/seed_42_segment_accuracy_comparison.csv`.
This table covers one outer split; do not treat its ranking as a stable result
without repeating the comparison on other splits.

For the predefined repeated comparison, run the same command with seeds 43,
44, 45, and 46. Each run skips completed distributions and resumes partial
histories. After all four distributions are complete for every seed, generate
the per-seed and paired summaries with:

```bash
python summarize_grouped_priors.py --seeds 42 43 44 45 46
```

The summary files are written to
`results/06_grouped_prior_comparison/segment_accuracy_repeated`.
The reported t intervals are descriptive because repeated outer splits overlap;
they are not confidence bounds from independent external cohorts. Choosing a
distribution after viewing these held-out results also makes this an exploratory
comparison rather than an unbiased final performance estimate.

## Jupyter Examples

```python
%run baseline_manual.py
%run bo_optimizer.py
%run random_search.py
%run bo_best_evaluation.py
%run compare_results.py
%run plot_results.py
%run subset_bo_experiments.py
%run make_subset_table3_metrics.py
%run plot_subset_results.py
%run meeting_results_summary.py
```

To only view saved main results without training:

```python
%run show_current_results_only.py
```

## Main Method

```text
parameters -> band-pass filtering -> window segmentation
-> PSD/statistical/entropy/Hjorth feature extraction
-> train/validation/test split
-> StandardScaler fitted on training set only
-> RBF-SVM
-> evaluation
```

Training set trains the SVM. Validation set is used by BO and Random Search to
select parameters. Test set is used only for final evaluation after parameters
are selected.

## Current BO Search Space

Fixed:

```text
class_weight = balanced
feature_setting = entropy_hjorth
```

BO tunes:

```text
f_low
f_high
window_size
overlap
C
gamma
```

The fixed feature setting means:

```text
PSD/statistical features
+ Shannon entropy
+ spectral entropy
+ approximate entropy
+ Hjorth activity, mobility, and complexity
```

The BO objective score is:

```text
0.35 * validation_accuracy
+ 0.25 * validation_macro_recall
+ 0.20 * validation_recall_N
+ 0.20 * validation_recall_F
```

This keeps the method simple while still reflecting the project finding that N
and F are the most confused classes.

## Metrics

The main five-class comparison reports:

```text
test_accuracy
test_macro_recall
test_recall_Z, test_recall_O, test_recall_N, test_recall_F, test_recall_S
test_seizure_sensitivity
confusion_matrix
```

For Table 3 style subset tasks, report:

```text
Case, Accuracy, Sensitivity, Specificity, Precision, Recall
```

Do not report F-measure, BO iteration, or Feature Setting in the final table.

## Output Folders

Main direct BO-SVM outputs:

```text
results/00_main_bo_svm
```

Subset full BO outputs:

```text
results/01_subset_tasks
```
