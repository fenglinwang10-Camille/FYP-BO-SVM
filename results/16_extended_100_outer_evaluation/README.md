# Outer evaluation of frozen 100-candidate selections

Run `D:\Anaconda3\envs\eegbo\python.exe -u -B evaluate_extended_outer.py` from the project root.

This supplementary evaluation uses experiment 14's completed 100-evaluation searches, which continued experiment 13's decision-score tie-breaking protocol. It is separate from experiment 10's original lowest-index vote rule. No search is run here.

Before prediction, the evaluator verifies source hashes, library versions, dataset hash, the five saved splits, all 100 history rows per method/fold, and earliest-best selected parameters. All five folds' selections are frozen together before evaluation.

Each changed selected configuration is refitted on 400 original outer-training segments and evaluated on its 100 outer-test segments using the same voting rule as the inner search. Unchanged selected parameters (including the fixed reference) reuse experiment 13's predictions after checking the source result signature; per-result provenance records this reuse. These are not new independent test observations.

Outputs include per-fold result JSON files, all 500 out-of-fold predictions per method, pooled confusion matrices and recalls, fold accuracy mean/sample SD, and paired BO-minus-RS differences. Original search and 50-evaluation result files are not overwritten.

The dataset and prior test results were inspected during development. The budget extension was prompted by observed search histories. These results are an exploratory reanalysis, not untouched or external validation. Neither test accuracy nor differences between protocols should be used to choose which experiment to report as if prespecified.
