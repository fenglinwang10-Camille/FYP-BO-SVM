# db4 decomposition depth: paired training-only comparison

Run `D:/Anaconda3/envs/eegbo/python.exe compare_wavelet_levels.py` from the
project root. The script resumes candidate histories with a matching protocol.

This developmental screen compares level 3 (53 features) and level 4 (58
features). Both remove original zero-based indices 15 and 32 (near-duplicate
gamma relative power and duplicate Hjorth activity). The common non-wavelet
block contains 33 features. Each wavelet coefficient group contributes energy,
relative energy, coefficient entropy, mean and standard deviation. Level 3
uses A3,D3,D2,D1; level 4 uses A4,D4,D3,D2,D1. Hence this compares complete
representations, not decomposition depth at a fixed feature dimension.

PyWavelets uses db4 and its default symmetric extension. The script checks
level feasibility even at the shortest 1-second window allowed by the main
search, and checks exact numerical compatibility of level 3 with the existing
deduplicated extractor. Wavelet bands are approximate dyadic frequency bands,
not exact conventional delta/theta/alpha/beta/gamma bands.

The five saved outer-training splits (seeds 42--46) and their recording-level
CV folds are reused. Filtering is fixed at 1--40 Hz and nonoverlapping windows
at 5 seconds. Both depths receive the same 50 SVM parameter candidates per
seed: reference C=10, gamma=.01 and 49 log-uniform random pairs over
C=[.1,100], gamma=[.0001,1]. Fold-local StandardScaler and RBF-SVM match the
main pipeline. The objective is mean fold recording accuracy after majority
vote. Each depth selects its own candidate; score ties retain the first.

No outer-test features or scores are computed. Selected CV scores reuse the
tuning folds and are optimistic; outer-training splits overlap. Results are
descriptive, not independent confirmation or a significance test. They apply
to the fixed preprocessing setting, not all window/filter configurations.
No main experiment or production feature extractor is changed.

Outputs include full histories, selected CV recording predictions, per-seed
scores, mean/SD summaries, paired differences, protocol/source hashes and
PNG/PDF plots. Plot shading is descriptive +/- one sample SD across five
splits. The runner validates paired candidates and training prediction IDs.

## Completed results (2026-11-05)

All 500 candidate evaluations completed. Selected training-CV accuracy:

| db4 depth | Features | Mean accuracy | SD (percentage points) |
| --- | --- | --- | --- |
| 3 | 53 | 88.20% | 1.02 |
| 4 | 58 | 87.15% | 1.47 |

Depth 4 minus depth 3 averaged -1.05 percentage points; depth 4 scored higher
on one split and lower on four. These exploratory results do not support
replacing the existing level-3 representation with level 4. No claim is made
that level 3 is optimal across other preprocessing settings, datasets or
search budgets. Retain depth 3 for the next planned experiment rather than
expanding the search based on these limited results.
