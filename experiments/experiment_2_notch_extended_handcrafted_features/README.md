# Experiment 2: Notch + Extended Handcrafted Features

This directory contains a fully independent Bonn EEG BO-SVM experiment.

Key differences from the earlier 35-feature experiment:

- fixed 50 Hz zero-phase notch filtering before band-pass filtering;
- extended preprocessing search space;
- extended non-wavelet handcrafted features;
- segment-level evaluation as the primary reporting level;
- separate outputs under `results/`, `figures/`, `cached_features/`, and `logs/`.

The main runner is:

`experiment_2_notch_extended_handcrafted_features_runner.py`
