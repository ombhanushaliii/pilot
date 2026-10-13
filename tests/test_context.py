import math
import unittest
import numpy as np
import torch

from cf_pilot.benchmark import FEATURE_COLUMNS, FEATURE_GROUPS, behavior_features
from cf_pilot.context import compact_history, history_features, HISTORY_COLUMNS, CONTEXT_COLUMNS
from cf_pilot.outcomes import audit_outcomes
from cf_pilot.recommendation_rl import (Network, NFEATURES, observations, infer,
                                       rank, learn, measure)
from tools.test_recommendations import fit_preprocessor, transform, candidate_data


def example():
    row = dict.fromkeys(FEATURE_COLUMNS, 0.)
    row.update(attempts=4, attempted_problems=2, first_submission_fraction=.1,
               last_submission_fraction=.8, gap_median_fraction=.2, retry_fraction=.5)
    return row


class ContextTests(unittest.TestCase):
    def test_normalization_uses_training_values_only(self):
        raw = np.zeros((2, NFEATURES)); raw[:, 0] = [2, 4]
        pre = fit_preprocessor(raw)
        future = np.full((1, NFEATURES), 10000.)
        transformed = transform(future, pre)
        self.assertEqual(pre["stats"][0]["mean"], 3.)
        self.assertEqual(pre["stats"][0]["scale"], 1.)
        self.assertEqual(transformed[0, 0], 9997.)

    def test_baseline_zero_placeholders_remove_history_and_its_missing_flags(self):
        x = np.ones((2, 2 * NFEATURES), np.float32)
        original = {"x": x, "y": np.array([0, 1])}
        changed = candidate_data(original, {"history": False})
        self.assertTrue((changed["x"][:, :24] == 1).all())
        self.assertTrue((changed["x"][:, 24:NFEATURES] == 0).all())
        self.assertTrue((changed["x"][:, NFEATURES + 24:] == 0).all())
        self.assertTrue((original["x"] == 1).all())

    def test_raw_history_callback_keeps_outcome_excluded_single_problem_sessions(self):
        contest = {"id": 77, "name": "fixture", "startTimeSeconds": 1770000000, "durationSeconds": 600}
        submission = {"id": 1, "contestId": 77, "creationTimeSeconds": 1770000001,
                      "problem": {"index": "A"}, "programmingLanguage": "C++17",
                      "author": {"members": [{"handle": "a"}], "participantType": "CONTESTANT"},
                      "verdict": "SKIPPED"}
        captured = []
        _, records = audit_outcomes([submission], contest, True, year=2026,
                on_session=lambda handle, items: captured.append(behavior_features(items, contest)))
        self.assertEqual(records[0]["pattern"], "all_skipped_single_problem")
        self.assertEqual(len(captured), 1)
        normal = {**submission, "verdict": "OK"}
        self.assertEqual(captured[0], behavior_features([normal], contest))

    def test_no_history_is_explicit_and_does_not_fake_personal_baseline(self):
        out = history_features(example(), [], 200)
        self.assertEqual(out["history_count_log"], 0.)
        self.assertTrue(all(out[n] is None for n in HISTORY_COLUMNS[1:]))

    def test_temporal_boundary_rejects_same_or_future_contests(self):
        for end in [200, 201]:
            with self.assertRaises(ValueError):
                history_features(example(), [compact_history(example(), end)], 200)

    def test_history_ignores_verdict_identity_and_administrative_fields(self):
        clean = example(); other = {**clean, "verdict": "SKIPPED", "handle": "ignored", "participantType": "OUT_OF_COMPETITION"}
        self.assertEqual(compact_history(clean, 100), compact_history(other, 100))
        out = history_features(other, [compact_history(clean, 100)], 200)
        self.assertAlmostEqual(out["history_count_log"], math.log(2))
        self.assertEqual(out["attempts_history_log_deviation"], 0.)

    def test_single_attempt_history_is_retained_with_missing_gaps(self):
        prior = example(); prior["attempts"] = 1; prior["attempted_problems"] = 1; prior["gap_median_fraction"] = None
        out = history_features(example(), [compact_history(prior, 100)], 200)
        self.assertGreater(out["history_count_log"], 0.)
        self.assertIsNone(out["history_gap_log_median"])
        self.assertIsNone(out["gap_history_log_deviation"])

    def test_hidden_groups_stay_hidden_but_cache_history_is_initially_visible(self):
        a = np.zeros((1, 2 * NFEATURES), np.float32)
        b = a.copy(); b[:, 5:24] = 1; b[:, NFEATURES + 5:NFEATURES + 24] = 1
        np.testing.assert_array_equal(observations(a, [0]), observations(b, [0]))
        b[:, 24] = 1
        self.assertFalse(np.array_equal(observations(a, [0]), observations(b, [0])))

    def test_current_gap_history_deviation_cannot_bypass_pacing_inspection(self):
        first = example(); second = {**first, "gap_median_fraction": .3}
        prior = [compact_history(first, 100)]
        x = np.zeros((2, 2 * NFEATURES), np.float32)
        for i, row in enumerate([first, second]):
            combined = {**row, **history_features(row, prior, 200)}
            x[i, :NFEATURES] = [combined[name] for name in CONTEXT_COLUMNS]
        np.testing.assert_array_equal(observations(x[:1], [0]), observations(x[1:], [0]))
        self.assertFalse(np.array_equal(observations(x[:1], [1]), observations(x[1:], [1])))
        retry = CONTEXT_COLUMNS.index("retry_history_deviation")
        changed = x[:1].copy(); changed[:, retry] += 1
        np.testing.assert_array_equal(observations(x[:1], [0]), observations(changed, [0]))
        self.assertFalse(np.array_equal(observations(x[:1], [2]), observations(changed, [2])))

    def test_zero_history_controls_are_equivalent_to_old_visibility_for_every_mask(self):
        x = np.random.default_rng(42).normal(size=(3, 2 * NFEATURES)).astype(np.float32)
        x[:, 24:NFEATURES] = 0; x[:, NFEATURES + 24:] = 0
        for mask in range(8):
            selected = list(range(24, NFEATURES)) + list(range(5))
            for i, group in enumerate(["pacing", "trajectory", "language"]):
                if mask & (1 << i):selected += [FEATURE_COLUMNS.index(name) for name in FEATURE_GROUPS[group]]
            visible = np.zeros(2 * NFEATURES, np.float32)
            visible[selected] = 1; visible[[i + NFEATURES for i in selected]] = 1
            bits = np.tile([(mask >> i) & 1 for i in range(3)], (3, 1)).astype(np.float32)
            expected = np.concatenate([np.clip(x, -8, 8) * visible, bits,
                           (3 - bits.sum(axis=1))[:, None] / 3], axis=1)
            np.testing.assert_array_equal(observations(x, [mask]*3), expected)

    def test_full_visible_controls_are_unchanged_by_derived_feature_grouping(self):
        x = np.random.default_rng(13).normal(size=(3, 2 * NFEATURES)).astype(np.float32)
        expected = np.concatenate([np.clip(x, -8, 8), np.ones((3,3)), np.zeros((3,1))],axis=1).astype(np.float32)
        np.testing.assert_array_equal(observations(x, [7]*3), expected)

    def test_all_visible_reference_has_only_terminal_actions(self):
        net = Network()
        with torch.no_grad():
            for parameter in net.parameters(): parameter.zero_()
        candidate = {"full_visible": True}
        pred, inspected = infer(net, np.zeros((2, 2 * NFEATURES), np.float32), candidate)
        np.testing.assert_array_equal(pred, [0, 0])
        np.testing.assert_array_equal(inspected, [0, 0])

    def test_contest_selection_is_not_dominated_by_larger_contest(self):
        a = {"macro_f1": .5, "worst_contest_f1": .4, "pooled": {"f1": .4, "precision": .5}}
        b = {"macro_f1": .4, "worst_contest_f1": .1, "pooled": {"f1": .8, "precision": .9}}
        self.assertGreater(rank(a), rank(b))
        self.assertLess(rank(a, True), rank(b, True))

    def test_full_visible_q_learner_learns_reward_task(self):
        x = np.zeros((2, 2 * NFEATURES), np.float32); x[:, 5] = [-1, 1]
        data = {"x": x, "y": np.array([0, 1], np.int8), "cid": np.array([1, 1])}
        settings = {"width": 32, "updates": 300, "validation_interval": 100, "environments": 8,
                    "learning_rate": .003, "target_tau": .05, "replay_per_class": 300,
                    "batch_size": 32, "warmup_updates": 20, "false_positive_penalty": .5}
        candidate = {"name": "fixture", "full_visible": True, "cost": 0.}
        saved = learn(data, data, candidate, settings, 42, lambda *a, **k: None)
        network = Network(32); network.load_state_dict(saved["macro"]["weights"])
        pred, inspected = infer(network, x, candidate)
        np.testing.assert_array_equal(pred, [0, 1])
        np.testing.assert_array_equal(inspected, [0, 0])
