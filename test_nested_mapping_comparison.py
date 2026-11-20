import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import nested_mapping_comparison as experiment
from nested_bo_rs import PARAMETERS, random_points
from parameter_priors import PREPROCESSING_BOUNDS


class NestedMappingTests(unittest.TestCase):
    def test_initial_quantiles_and_svm_parameters_match_reference(self):
        points = np.asarray(experiment.initial_coordinates(42))
        random = np.random.default_rng(42).random((50, 5))
        np.testing.assert_array_equal(points[:, :3], random[:16, :3])
        np.testing.assert_array_equal(points[:, 3:], np.asarray(random_points(42, 50))[:16, 3:])
        uniform = [experiment.physical_point(point, 'uniform') for point in points]
        np.testing.assert_allclose(uniform, random_points(42, 50)[:16], rtol=0, atol=1e-12)

    def test_mappings_are_monotone_bounded_and_keep_svm_parameters(self):
        for mapping in experiment.MAPPINGS:
            mapped = np.array([experiment.physical_point([u,u,u,10,.01], mapping)
                               for u in np.linspace(0,1,101)])
            for column, bounds in enumerate(PREPROCESSING_BOUNDS.values()):
                self.assertTrue(np.all(np.diff(mapped[:, column]) > 0))
                self.assertTrue(np.all(mapped[:, column] >= bounds.low))
                self.assertTrue(np.all(mapped[:, column] <= bounds.high))
            np.testing.assert_array_equal(mapped[:, 3:], np.tile([10,.01], (101,1)))

    def test_restart_reproduces_full_sequence_and_evaluation_budget(self):
        split = {'outer_fold':1, 'inner_folds':[
            {'train_segment_ids':[1,2], 'validation_segment_ids':[3]}]}
        protocol = dict(evaluations_per_method=4, initial_evaluations=2,
                        search_seed=42, xi=.01, parameter_mapping='normal')
        calls = []

        def evaluate(train_data, folds, physical, iteration, phase):
            self.assertEqual(train_data[2], [1,2,3])
            calls.append(physical)
            return dict(iteration=iteration, phase=phase, candidate_seconds=0.,
                        score=round(.75+physical[0]*.02, 2),
                        **dict(zip(PARAMETERS, physical)))

        with tempfile.TemporaryDirectory(dir=experiment.PROJECT_DIR/'tmp') as name:
            resumed, full = Path(name)/'resumed', Path(name)/'full'
            resumed.mkdir(); full.mkdir()
            with patch.object(experiment, '_evaluate_candidate', side_effect=evaluate):
                _, complete = experiment.run_search((None,None,[1,2,3]), split, protocol, resumed, max_new=3)
                self.assertFalse(complete)
                state, complete = experiment.run_search((None,None,[1,2,3]), split, protocol, resumed)
                self.assertTrue(complete)
                self.assertEqual(len(calls), 4)
                experiment.run_search((None,None,[1,2,3]), split, protocol, resumed)
                self.assertEqual(len(calls), 4)
                expected, _ = experiment.run_search((None,None,[1,2,3]), split, protocol, full)
            cols = (*PARAMETERS, *experiment.LATENT_NAMES, 'score')
            np.testing.assert_array_equal([[r[c] for c in cols] for r in state['history']],
                                           [[r[c] for c in cols] for r in expected['history']])


if __name__ == '__main__':
    unittest.main()
