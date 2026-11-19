# Bonn EEG BO-SVM experiments

Install the pinned dependencies with `python -m pip install -r requirements.txt`.
The dataset contains 500 original recordings in `Dataset/`.

## Nested evaluation with shared initialization

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nested_experiment.ps1
```

This starts or resumes the outer five-fold/inner five-fold comparison in the
background. BO and RS share their first 16 candidates and CV scores; each uses
50 candidates. The fixed reference is evaluated on the same outer folds.
See `README_NESTED_EXPERIMENT.md` for seeds, checkpoints and output files.
The running experiment saves under `results/10_nested_cv_shared_initialization/`.

## Earlier exploratory repeated-holdout experiment

```bash
python run_grouped_segment_priors.py --seed 42
python summarize_grouped_priors.py --seeds 42 43 44 45 46
python grouped_method_comparison.py --seed 44
```

The four mappings have completed results for seeds 42-46. For a new exploratory
run, pass the intended seed to the first command. The comparison requires the
completed uniform BO run for the same seed. These results use 55 features,
recording-fixed five-fold CV, and recording accuracy after majority voting.
They are saved under `results/09_fixed_fold_segment_accuracy/` and remain
separate from the nested evaluation. Seeds change the holdout split and search;
these overlapping holdouts are not independent datasets. Previously inspected
test results must not guide new mapping or configuration choices.

## Saved-result analysis

```bash
python tmp/analyze_search_behaviour.py
python tmp/build_chapter5_seed44_figures.py
python plot_grouped_parallel_coordinates_v2.py --seed 44
python plot_grouped_surrogate.py --seed 42 --prior uniform
python analyze_grouped_voting.py
```

The voting script is a training-CV-only exploratory tie-break analysis and does
not alter the main experiment's vote rule. `analyze_grouped_features.py` is
retained because the voting analysis imports its saved-fold loader. Its unused
feature-screen outputs were removed. Training-only PSD analysis and protocol
audits remain in `chapter4_draft/`.

Old validation protocols, subset tasks, alternative feature-selection and
53-feature experiments, recording averaging, soft-voting trials and notch
experiments were removed with their results and dedicated scripts. Presentation,
manuscript and reference-paper files are excluded from this experiment project.
