"""Protocol, shared-budget and restart checks without a full EEG search."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import nested_bo_rs as experiment


class NestedProtocolTests(unittest.TestCase):
    def test_every_recording_is_held_out_once_and_inner_folds_exclude_it(self):
        labels = np.repeat(np.arange(5), 100)
        splits = experiment.make_splits(labels)
        tests = [set(f["test_segment_ids"]) for f in splits]
        self.assertEqual(set.union(*tests), set(range(500)))
        self.assertEqual(sum(map(len, tests)), 500)
        for split in splits:
            training = set(split["train_segment_ids"])
            testing = set(split["test_segment_ids"])
            self.assertEqual(len(training), 400)
            self.assertFalse(training & testing)
            coverage = []
            for inner in split["inner_folds"]:
                fit = set(inner["train_segment_ids"])
                valid = set(inner["validation_segment_ids"])
                self.assertEqual(len(fit), 320)
                self.assertEqual(len(valid), 80)
                self.assertEqual(fit | valid, training)
                self.assertFalse(fit & valid)
                coverage.extend(valid)
            self.assertEqual(sorted(coverage), sorted(training))

    def test_shared_initial_scores_are_evaluated_once_and_restart_is_exact(self):
        protocol = {"evaluations_per_method": 4, "shared_initial_evaluations": 2,
                    "search_seed": 42, "xi": .01,
                    "fixed_reference_parameters": experiment.MANUAL_PARAMS}
        split = experiment.make_splits(np.repeat(np.arange(5), 100))[0]
        calls = []

        def evaluate(train_data, folds, point, iteration, phase):
            calls.append(tuple(point))
            self.assertEqual(set(train_data[2]), set(split["train_segment_ids"]))
            self.assertEqual(len(folds), 5)
            # A coarse objective with ties exercises deterministic earliest selection.
            return dict(iteration=iteration, phase=phase, candidate_seconds=0.,
                        score=round(.75 + .02 * point[0], 2),
                        **dict(zip(experiment.PARAMETERS, map(float, point))))

        train_data = (None, None, np.asarray(split["train_segment_ids"]))
        with tempfile.TemporaryDirectory(dir=experiment.PROJECT_DIR / "tmp") as temp:
            resumed = Path(temp) / "resumed"
            full = Path(temp) / "full"
            resumed.mkdir()
            full.mkdir()
            with patch.object(experiment, "_evaluate_candidate", side_effect=evaluate):
                # Stop immediately after the first guided BO point, before its RS partner.
                _, complete = experiment.run_searches(train_data, split, protocol, resumed, max_new=3)
                self.assertFalse(complete)
                state, complete = experiment.run_searches(train_data, split, protocol, resumed)
                self.assertTrue(complete)
                self.assertEqual(len(calls), 6)  # 2 shared + 2 BO + 2 RS, not 8.
                selection = experiment.selected_configurations(state, protocol)
                for method in ("bo", "rs"):
                    self.assertEqual(len(state["shared"] + state[method]), 4)
                before = len(calls)
                experiment.run_searches(train_data, split, protocol, resumed)
                self.assertEqual(len(calls), before)
                uninterrupted, _ = experiment.run_searches(train_data, split, protocol, full)
            for method in ("shared", "bo", "rs"):
                columns = (*experiment.PARAMETERS, "score", "iteration")
                actual = [[r[c] for c in columns] for r in state[method]]
                expected = [[r[c] for c in columns] for r in uninterrupted[method]]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
            self.assertEqual(selection, experiment.selected_configurations(uninterrupted, protocol))
            import pandas as pd
            bo = pd.read_csv(resumed / "bo_history.csv")
            rs = pd.read_csv(resumed / "rs_history.csv")
            np.testing.assert_array_equal(bo.loc[:1, [*experiment.PARAMETERS, "score"]],
                                          rs.loc[:1, [*experiment.PARAMETERS, "score"]])

    def test_protocol_changes_cannot_overwrite_saved_experiment(self):
        with tempfile.TemporaryDirectory(dir=experiment.PROJECT_DIR / "tmp") as temp:
            path = Path(temp) / "protocol.json"
            experiment._immutable_json(path, {"search_seed": 42})
            experiment._immutable_json(path, {"search_seed": 42})
            with self.assertRaises(ValueError):
                experiment._immutable_json(path, {"search_seed": 43})
            self.assertEqual(json.loads(path.read_text()), {"search_seed": 42})

    def test_candidate_evaluation_uses_explicit_inner_training_ids(self):
        labels = np.repeat(np.arange(5), 100)
        split = experiment.make_splits(labels)[0]
        training = np.asarray(split["train_segment_ids"])
        groups = np.repeat(training, 2)
        y = labels[groups]
        X = np.column_stack([groups, y])
        fitted_ids = []

        class Model:
            def fit(self, features, target):
                fitted_ids.append(sorted(set(features[:, 0].astype(int))))
            def predict(self, features):
                return features[:, 1].astype(int)

        folds = [(np.asarray(f["train_segment_ids"]), np.asarray(f["validation_segment_ids"]))
                 for f in split["inner_folds"]]
        point = [experiment.MANUAL_PARAMS[p] for p in experiment.PARAMETERS]
        with patch("grouped_bo_optimizer._build_feature_matrix", return_value=(X, y, groups)), patch(
            "grouped_bo_optimizer._make_model", side_effect=lambda params: Model()
        ):
            row = experiment._evaluate_candidate((None, labels[training], training),
                                                  folds, point, 1, "shared_initial")
        self.assertEqual(fitted_ids, [sorted(f["train_segment_ids"]) for f in split["inner_folds"]])
        self.assertEqual(row["score"], 1.)

    def test_outer_predictions_are_built_only_after_model_fit(self):
        events = []
        params = experiment.MANUAL_PARAMS
        train_ids = np.array([0, 1, 2, 3, 4])
        test_ids = np.array([10, 11, 12, 13, 14])

        def features(signals, labels, ids, configuration):
            events.append("train_features" if ids[0] == 0 else "test_features")
            return np.column_stack([labels]), np.asarray(labels), np.asarray(ids)

        class Model:
            def fit(self, X, y):
                events.append("fit")
            def predict(self, X):
                events.append("predict")
                return X[:, 0].astype(int)

        with patch.object(experiment, "_build_feature_matrix", side_effect=features), patch.object(
            experiment, "_make_model", return_value=Model()
        ):
            result = experiment.evaluate_outer((None, np.arange(5), train_ids),
                                                (None, np.arange(5), test_ids), params)
        self.assertEqual(events, ["train_features", "fit", "test_features", "predict"])
        self.assertEqual(result["segment_level_test_metrics"]["accuracy"], 1.)
        self.assertEqual([r["recording_id"] for r in result["recording_predictions"]], test_ids.tolist())


if __name__ == "__main__":
    unittest.main()
