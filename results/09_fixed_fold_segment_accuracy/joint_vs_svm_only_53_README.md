# Fixed preprocessing versus joint BO (53 features)

Run `D:/Anaconda3/envs/eegbo/python.exe compare_joint_preprocessing_bo.py --workers 2`.
Results are written into `joint_vs_svm_only_53/`; completed runs are skipped
and incomplete runs resume their saved GP optimizer and random state.

Both arms use the deduplicated 53-feature db4-level-3 representation, identical
saved recording folds per seed, RBF-SVM, fold-local scaling, GP/EI and 50
evaluations including 16 initialization points. Seeds are 42--46. The first
full candidate is identical; the remaining 15 initial C/gamma pairs are also
matched, with uniform preprocessing candidates for the joint arm.

SVM-only fixes filtering at 1--40 Hz and windows at 5 seconds, the reference
configuration from the earlier feature screens. This is a predeclared project
reference, not a literature-established optimum. Joint BO searches filtering
and window bounds from the existing experiment along with C/gamma. The two
arms have different search dimensions, so this compares complete strategies
under equal evaluation budgets, not an isolated causal effect of preprocessing.

After each run selects its highest mean fold recording accuracy, the selected
configuration is fitted on the outer training recordings and evaluated on the
saved outer test recordings. Test scores are not provided to BO. These splits
have been examined in earlier development, so this is not a pristine independent
validation; repeated splits overlap. No changes will be selected using the
test results from this comparison.

Per-candidate CSVs, atomic optimizer checkpoints, split/fold manifests,
selected parameters and per-record test predictions are saved. The default
55-feature production pipeline and its results are unchanged. Timing includes
search overhead but varies with parallel resource contention; evaluation count
is the primary budget. Summarize completed runs with `--summarize-only`.
