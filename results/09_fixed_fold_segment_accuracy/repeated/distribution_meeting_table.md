# Distribution comparison for meeting

Five-class Bonn EEG classification; segment_accuracy objective; five matched
outer train-test splits (seeds 42-46); 50 BO evaluations per run.

| BO parameter-search setting | Test segment accuracy (mean +/- SD) |
| --- | ---: |
| uniform | **86.8% +/- 3.0%** |
| normal | 84.8% +/- 2.8% |
| exponential | 84.4% +/- 3.6% |
| gamma | 85.0% +/- 2.0% |

These distributions map the BO search coordinates for f_low, f_high and
window_size. They are not distributions fitted to the EEG recordings.
Uniform has the highest descriptive mean, but paired accuracy confidence
intervals against uniform include zero; superiority is not established.

Source: summary_by_prior.csv (rows where metric = segment_accuracy).

## Seed 42: same outer train-test split

| BO parameter-search setting | Held-out test segment accuracy |
| --- | ---: |
| uniform | **90%** |
| normal | 86% |
| gamma | 85% |
| exponential | 81% |

Each percentage is the proportion of 100 held-out EEG recordings classified
correctly after window-level majority voting. The four runs share the same
seed-42 outer split and use the segment_accuracy objective and 50 BO calls.
Uniform is best on this one split; the five-split summary above is the more
appropriate overall result.

Source: per_seed.csv (rows where seed = 42).
