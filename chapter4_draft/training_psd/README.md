# Training-recording PSD analysis

Created for Chapter 4 on 2026-11-08. This is a descriptive analysis added after the main experiment, not evidence of how the original search bounds were chosen.

## Existing analysis audit

`plot_unfiltered_average_psd.py` loads all 500 recordings without applying a train/test split. Its outputs in `results/raw_psd_inspection` therefore describe the complete dataset, not only the training data. The separate notch experiment also contains all-recording PSD diagnostics. Neither is labelled as a training-only figure here.

## New analysis

The saved seed-42 main-experiment manifest determines the 400 training recording IDs (80 per class). The script matches the main loader's sorted file order, validates 500 recognised files and disjoint train/test IDs, and reads signal values only for the selected training files. The included recording list and file hashes make the input selection auditable.

Welch PSD is calculated on each unfiltered full recording, using fs=173.61 Hz, a 1024-sample Hann window, 512-sample overlap, constant detrending, and density scaling. The frequency spacing is approximately 0.16954 Hz. Arithmetic averages are computed within each class before plotting on a logarithmic vertical axis. No band-pass or notch filter and no amplitude normalisation is applied. The y-axis uses squared recorded signal units per Hz rather than assuming a physical unit. The 1024-sample PSD setting is for this full-recording illustration; classification features continue to use min(256, window length).

## Observations

- Integrating the class-mean spectra shows that 0.5–30 Hz accounts for approximately 88.5%, 95.6%, 92.8%, 93.0%, and 99.0% of total estimated power for Z, O, N, F, and S respectively. These are fractions of class-mean PSD power, not means of individual recording fractions.
- The O mean spectrum has a clear peak near 11.19 Hz. The S spectrum has its highest peak within 0.5–40 Hz near 5.26 Hz and greater absolute power over much of the low-frequency range.
- Narrow peaks near 50 Hz are visible. Their origin cannot be established from this plot alone. A band-pass upper cutoff above 50 Hz does not necessarily remove these peaks.
- Lower high-frequency power does not establish that those components are noise or uninformative for classification. Class-mean differences do not prove that individual records can be separated accurately.

## Paper use

Place the figure near the beginning of the preprocessing subsection. Describe the observed frequency distribution, then explain that validation is used to assess the classification consequences of changing filter cutoffs. Do not claim that this analysis proves the exact bounds or an optimal band. No search settings, feature extraction, or classification results were changed.

Files: `training_class_mean_psd.pdf` (vector figure), `.png` (preview), `figure_insert.tex` (caption and insertion), `class_mean_psd.csv`, `per_recording_psd.csv`, `spectral_summary.csv`, `training_recordings.csv`, and `protocol.json`.

Reproduce from the project root using the eegbo Python environment and `chapter4_draft/analyse_training_psd.py`.
