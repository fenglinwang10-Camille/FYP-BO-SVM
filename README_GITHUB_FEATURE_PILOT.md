# Training-only GitHub feature pilot

This exploratory screen compares the existing 55 feature columns with the
upstream **catch24** definition: the 22 canonical catch22 features plus the raw
window mean and standard deviation. It is a generic time-series representation,
not an EEG-specific model or a reproduction of a claimed Bonn accuracy.

Sources:

- https://github.com/DynamicsAndNeuralSystems/pycatch22
- https://github.com/DynamicsAndNeuralSystems/catch22
- Lubba et al. (2019), *catch22: CAnonical Time-series CHaracteristics*,
  https://doi.org/10.1007/s10618-019-00647-x
- An alternative EEG-specific resource inspected during selection was
  https://github.com/mne-tools/mne-features. Its Bonn demonstration is binary
  seizure detection with four features and a random forest; its scores are not
  directly comparable to this project's five-class SVM task.

## Frozen pilot design

Only the 400 outer-training recordings of outer fold 1 are opened. The original
100 outer-test signals and saved outer predictions are never opened. The saved
five inner folds (320 fit / 80 validate) are reused unchanged. All windows of a
recording remain together. The 500 file paths are enumerated only to preserve
the original global recording IDs.

Both versions use the same 0.5–40 Hz Butterworth filter, 2 s non-overlapping
windows, per-fold StandardScaler and RBF-SVM, and recording majority voting.
Each evaluates the same 12 coarse grid points: C in {1,10,100} and gamma in
{0.0001,0.001,0.01,0.1}. Feature matrices are extracted once per version. This
tests a small fixed preprocessing setting, not either representation's optimum
over the full five-parameter space. This stage is not a BO/RS comparison.

The primary pilot score is mean inner-fold **recording** accuracy. Per-class
recall, extraction time, and SVM search time are saved. The highest inner score
has selection optimism and is not an estimate of new-data performance. No outer
evaluation or automatic promotion to the main experiment is implemented.

Existing main experiment source files, results and background jobs are untouched.
The installed dependency is isolated under `tmp/github_feature_deps`; no
package in the original environment is upgraded. The upstream package retains
its GPL licensing and is imported, not copied into project source code. Record
the installed version from the saved protocol when reporting any results.

## Run

From the project root, with the existing EEG Python environment:

```powershell
python -m pip install pycatch22 --no-deps --target tmp/github_feature_deps
python -B pilot_github_features.py --smoke
python -u -B pilot_github_features.py
```

The smoke check verifies finite, deterministic catch24 extraction and the
existing 55-column shape on five training windows. The pilot also checks every
inner fold against the allowed training recording IDs. The output directory is
`results/12_training_feature_pilot/catch24_outer_fold_1`. Existing protocols are
not overwritten. See `status.json`, `run.log`, `history.csv`, and `summary.csv`.

Saved CSV/JSON results are versioned; regenerable feature matrices and local
logs are excluded. After cloning on another device, run
`python -B rebuild_training_feature_cache.py` before using the follow-up
basic23 or voting scripts. This only opens the 400 saved training files and
does not require pycatch22. The original catch24 run can be reproduced in a
fresh output directory using the pinned `pycatch22==1.0.2` dependency; the
existing immutable result directory is intentionally protected from overwrite.

## Completed first pilot

Both feature sets selected C=10, gamma=0.1 from the same 12-point grid.

| Representation | Columns | Best mean inner-CV recording accuracy | Extraction time | Twelve-point search time |
|---|---:|---:|---:|---:|
| original_55 | 55 | 85.25% | 39.86 s | 35.78 s |
| catch24 | 24 | 84.50% | 5.02 s | 28.78 s |

Catch24's selected inner-CV recall for F was 67.50%, compared with 62.50%
for original_55; N recall decreased from 82.50% to 76.25%. Thus this pilot
does not show an overall accuracy improvement. Catch24 extracted more quickly
in this particular run, but concurrent machine activity affects timing.
No outer test evaluation was performed. These numbers cannot be compared
directly against the previous five-parameter, 50-evaluation BO search because
this pilot fixes filtering and window length and tunes only C and gamma.

## Basic 23-feature follow-up

`pilot_basic23_features.py` runs the existing `original` feature setting:
five absolute band powers, five relative band powers and thirteen statistics.
It uses exactly the previous pilot's saved training files, inner folds,
preprocessing and twelve SVM configurations. Existing 55-dimensional results
are reused rather than rerun. Before model fitting, the new 23 columns are
checked against the corresponding columns of the saved 55-feature matrix.
The main experiment and the first pilot sources remain unchanged.

Results are saved separately in
`results/12_training_feature_pilot/basic23_outer_fold_1`.
The best mean inner-CV accuracy was 80.50% (C=10, gamma=0.1), compared with
85.25% for the existing 55 features and 84.50% for catch24. Feature extraction
took 7.14 seconds; the twelve-point SVM search took 31.49 seconds. F-class
recall was 43.75%, versus 62.50% with 55 features. Thus this limited screen
does not support replacing the main representation with the basic 23 features.
All figures remain inner-validation results from one outer training set;
no external or outer-test performance has been estimated for these alternatives.

## Recording aggregation pilot

`pilot_voting_rules.py` reuses the cached original 55-dimensional training
features, all twelve SVM configurations and the five saved inner folds. It fits
one SVM per configuration/fold, then compares the same window outputs under:

1. Majority vote with the lowest class index on ties (the existing rule).
2. Majority vote, resolving tied classes by their mean OVR decision scores.
3. The class with the highest mean OVR decision score over all windows.

OVR decision scores are not calibrated probabilities. Rule 2 only chooses among
classes tied in recording vote counts. If mean scores also tie, the lowest
class index is retained. These are recording aggregation rules; they do not
change SVC's internal window prediction tie handling (`break_ties`).

The pilot checks a 2-2-1 multiclass vote tie and a unique vote winner before
running, and reproduces the original majority-vote fold accuracies for all
twelve configurations. Results are written to
`results/12_training_feature_pilot/voting_rules_outer_fold_1`.
`best_per_rule.csv` reports the best configuration for each rule, while
`same_configuration_comparison.csv` compares all rules at the configuration
chosen by the original voting rule. `inner_recording_predictions.csv` provides
the 400 inner-validation predictions per configuration for auditing changes.
No outer-test signals or saved outer-test predictions are read.

All three rules selected C=10, gamma=0.1 in this pilot. Mean inner accuracy was
85.25% for the original rule, 86.00% for decision-score tie-breaking, and 85.75%
for mean-score aggregation. The selected configuration had eleven vote-tied
recordings. Tie-breaking changed ten predictions, corrected five errors and
introduced two errors, a net gain of three out of 400 inner-validation
recordings. F recall increased from 62.50% to 67.50%, while N recall fell from
82.50% to 81.25%. Mean-score aggregation produced F recall of 70.00% and N
recall of 77.50%. These small training-validation improvements do not establish
an outer-test benefit or justify rewriting the completed main experiment.
