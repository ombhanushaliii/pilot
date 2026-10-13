import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cf_pilot.benchmark import FEATURE_COLUMNS, file_hash
from cf_pilot.rl import REWARDS, TabularAgent
from cf_pilot.training import metrics, rollout_batch, scores, select_threshold, sliced_metrics, train_agent, train_logistic


CONFIG = {"rewards": REWARDS, "inspection_cost": .05, "budget": 3,
          "episodes": 500, "checkpoint_interval": 100, "alpha": .1, "gamma": 1.,
          "epsilon_start": 1., "epsilon_end": .05, "epsilon_decay_episodes": 400,
          "seeds": [42, 736], "logistic_c": [.1, 1.]}


class TrainingTests(unittest.TestCase):
    def test_unseen_user_slices_preserve_targets_and_handle_membership(self):
        data = {"y": np.array([0, 1, 1]), "metadata": [
            {"handle": "train", "contest_id": "1", "participation_stratum": "CONTESTANT"},
            {"handle": "validation", "contest_id": "1", "participation_stratum": "CONTESTANT"},
            {"handle": "unseen", "contest_id": "1", "participation_stratum": "OUT_OF_COMPETITION"}]}
        result = sliced_metrics(data, np.array([0, 1, 0]), np.array([0, 1, 2]), CONFIG, {"train"}, {"validation"})
        self.assertEqual(result["unseen_in_train"]["sessions"], 2)
        self.assertEqual(result["unseen_in_train_and_validation"]["sessions"], 1)
        self.assertEqual(result["unseen_in_train_and_validation"]["confusion"]["fn"], 1)

    def test_hand_calculated_metrics_with_asymmetric_rewards(self):
        out = metrics([1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 2, 3], CONFIG)
        self.assertEqual(out["confusion"], {"tp": 1, "fn": 1, "fp": 1, "tn": 1})
        self.assertEqual(out["precision"], .5)
        self.assertEqual(out["recall"], .5)
        self.assertAlmostEqual(out["mean_reward"], (20 - 20 - 2 + 1) / 4 - .075)

    def test_threshold_ties_grouped_and_all_negative_candidate(self):
        p = np.array([.9, .8, .8, .2])
        y = np.array([0, 1, 0, 0])
        threshold = select_threshold(p, y, CONFIG)
        self.assertEqual(threshold, .8)
        best = metrics(y, p >= threshold, 0, CONFIG)["mean_reward"]
        for trial in (.91, .9, .8, .2):
            self.assertGreaterEqual(best, metrics(y, p >= trial, 0, CONFIG)["mean_reward"])
        self.assertGreater(select_threshold(p, np.zeros(4), CONFIG), .9)

    def test_rollout_batch_matches_individual_and_ignores_targets(self):
        data = {"bins": [(1, 2, 3, 0, 1, 2), (1, 2, 3, 4, 4, 4)], "y": np.array([0, 1])}
        agent = TabularAgent("q_learning")
        first = rollout_batch(data, agent)
        data["y"] = np.array([1, 0])
        second = rollout_batch(data, agent)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_saved_logistic_coefficients_match_scores(self):
        workspace = Path(__file__).resolve().parents[1] / "work"
        with tempfile.TemporaryDirectory(dir=workspace) as temp:
            x = np.array([[-2.], [-1.], [1.], [2.]], dtype=np.float32)
            data = {"x": x, "y": np.array([0, 0, 1, 1])}
            path = Path(temp) / "logistic.json"
            trained = train_logistic(data, data, "fixture", [0], CONFIG, path)
            saved = json.loads(path.read_text())
            np.testing.assert_array_equal(scores(trained["model"], x), scores(saved["model"], x))

    def test_two_agents_complete_validation_checkpoint_selection(self):
        workspace = Path(__file__).resolve().parents[1] / "work"
        with tempfile.TemporaryDirectory(dir=workspace) as temp:
            data = {"bins": [(0, 0, 0, 1, 1, 1), (3, 3, 3, 1, 1, 1)], "y": np.array([0, 1])}
            for algorithm in ("q_learning", "sarsa"):
                path = Path(temp) / (algorithm + ".json")
                saved = train_agent(data, data, algorithm, 42, CONFIG, path, lambda *a, **k: None)
                self.assertEqual(len(saved["history"]), 5)
                self.assertEqual(saved["validation"]["mean_reward"], max(h["validation"]["mean_reward"] for h in saved["history"]))
                agent = TabularAgent.restore(saved["model"])
                pred, cost = rollout_batch(data, agent)
                np.testing.assert_array_equal(pred, [0, 1])
                self.assertTrue((cost <= 3).all())
