import numpy as np
import pandas as pd

from plot_grouped_parallel_coordinates import _normalise_history


def test_parallel_coordinates_use_log_scales_for_svm_parameters():
    frame = pd.DataFrame({
        "f_low": [0.5, 5.0],
        "f_high": [30.0, 60.0],
        "window_size": [1.0, 5.0],
        "C": [0.1, 100.0],
        "gamma": [1e-4, 1.0],
    })
    coordinates = _normalise_history(frame)
    np.testing.assert_allclose(coordinates[0], np.zeros(5))
    np.testing.assert_allclose(coordinates[1], np.ones(5))
    frame.loc[0, ["C", "gamma"]] = [np.sqrt(0.1 * 100), 0.01]
    coordinates = _normalise_history(frame)
    np.testing.assert_allclose(coordinates[0, 3:], [0.5, 0.5])
