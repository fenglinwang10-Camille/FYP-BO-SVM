# Meeting Comparison Summary

Source: `results\00_main_bo_svm\comparison_summary.csv`

Best method by test accuracy: **SVM with BO**

| Method | Test Accuracy | Recall Z | Recall O | Recall N | Recall F | Recall S | Seizure Sensitivity | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM without BO | 76.45% | 79.09% | 82.27% | 68.18% | 54.09% | 98.64% | 98.64% |  |
| SVM with BO | 86.00% | 93.12% | 84.38% | 86.88% | 66.25% | 99.38% | 99.38% | YES |
| Random Search SVM | 84.25% | 83.75% | 85.00% | 76.25% | 76.25% | 100.00% | 100.00% |  |

## Best Parameters

| Method | Key Parameters |
| --- | --- |
| SVM without BO | f_low: 0.5, f_high: 40.0, window_size: 2.0, C: 10.0, gamma: 0.1 |
| SVM with BO | f_low: 0.866676218816071, f_high: 30.0, window_size: 5.0, C: 9.941294181492, gamma: 0.0163193791142169, overlap: 0.5, class_weight: balanced, feature_setting: nf_enhanced |
| Random Search SVM | f_low: 4.296087282189888, f_high: 57.07959431782851, window_size: 4.918282722346371, C: 25.472856018371225, gamma: 0.1311928489172758, class_weight: balanced, feature_setting: entropy_hjorth |
