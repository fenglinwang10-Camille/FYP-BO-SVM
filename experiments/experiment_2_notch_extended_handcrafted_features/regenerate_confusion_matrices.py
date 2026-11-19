from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import experiment_2_notch_extended_handcrafted_features_runner as runner


def main() -> None:
    specs = [
        ("bo", "BO-SVM Segment-Level Confusion Matrix"),
        ("random", "Random Search Segment-Level Confusion Matrix"),
        ("manual", "Manual SVM Segment-Level Confusion Matrix"),
    ]
    for method, title in specs:
        df = pd.read_csv(ROOT / "results" / f"confusion_matrix_{method}_segment_level.csv")
        cm = df.drop(columns=["true_label"]).to_numpy(dtype=int)
        runner.plot_confusion_matrix(
            cm,
            title,
            ROOT / "figures" / f"confusion_matrix_{method}_segment_level.png",
        )
        print(method)


if __name__ == "__main__":
    main()
