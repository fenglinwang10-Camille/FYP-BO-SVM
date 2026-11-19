# Paired feature deduplication screen

Run `D:/Anaconda3/envs/eegbo/python.exe compare_feature_deduplication.py`
from the project root. The script resumes saved candidate evaluations.

This is a training-only developmental comparison, not a replacement for the
completed main BO experiments. It compares the original 55 columns against
53 columns obtained by removing zero-based indices 15 and 32. Index 15 is a
near-duplicate of relative gamma power (index 9); index 32 is Hjorth activity,
identical to variance (index 18). No other feature definitions are changed.

Each seed 42--46 reuses the saved outer-training recordings and five recording
folds. Preprocessing is fixed at 1--40 Hz, 5-second non-overlapping windows,
matching the earlier feature screen. For each seed, both representations
evaluate the same 50 SVM configurations: the reference C=10, gamma=0.01 plus
49 paired random configurations, log-uniform over C=[0.1,100] and
gamma=[0.0001,1]. Scaling is fitted separately inside each training fold.
The objective is the equally weighted mean fold recording accuracy after
window majority vote. Each representation selects its own best candidate;
ties retain the first candidate.

Only outer-training recordings are transformed and scored. The existing
loader reads the source dataset, but outer-test signals are not used in feature
extraction, fitting, candidate selection, or evaluation. No held-out test
scores are produced. Selected CV scores reuse tuning folds and are optimistic;
the overlapping outer-training splits are not independent samples. Differences
are descriptive and do not establish superiority or equivalence.

Search timings cover fold scaling, fitting, prediction and scoring, excluding
feature extraction and final prediction export. Both versions extract the
original 55 columns then slice, so this experiment cannot establish feature
extraction time savings. Existing production scripts continue using 55 features.

`protocol.json` records settings and source hashes; `*_history.csv` records
all candidates; `*_audit.json` records actual duplicate differences;
`per_seed.csv`, `summary.csv` and `paired_differences.csv` contain completed
comparisons. `*_selected_cv_predictions.csv` contains training-CV recording
predictions for the selected candidates, not independent test predictions.

## Completed comparison (2026-11-04)

All 500 candidate evaluations completed (5 splits x 2 versions x 50 settings).
Selected mean fold recording accuracy across splits:

| Version | Mean | SD |
| --- | --- | --- |
| Original 55 | 88.40% | 1.28 percentage points |
| Deduplicated 53 | 88.20% | 1.02 percentage points |

The mean paired change (53 minus 55) is -0.20 percentage points. This small
observed difference supports considering the nonredundant representation,
but does not prove equivalence or improved generalisation. This experiment
holds preprocessing fixed and does not replace a joint BO evaluation.
The production feature setting has not been changed. Main results must be
recomputed consistently if a new representation is adopted.

Run `D:/Anaconda3/envs/eegbo/python.exe plot_feature_deduplication.py` to
validate matched candidates, complete histories, selected scores and
training-only prediction IDs, and regenerate the PNG/PDF comparison plot.
