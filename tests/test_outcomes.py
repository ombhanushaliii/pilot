import unittest

from cf_pilot.outcomes import YEAR_START, YEAR_END, audit_outcomes, year_bounds


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.contest = {"id": 1, "name": "Test", "startTimeSeconds": YEAR_START + 100,
                        "durationSeconds": 7200}

    def sub(self, sid, verdict, problem="A", kind="CONTESTANT", created=None):
        return {"id": sid, "contestId": 1, "creationTimeSeconds": created if created is not None else YEAR_START + 100 + sid,
                "problem": {"index": problem}, "verdict": verdict,
                "author": {"participantType": kind, "members": [{"handle": "Alice"}]}}

    def check(self, items, complete=True):
        return audit_outcomes(items, self.contest, complete)

    def test_replacement_compatible_is_not_misconduct(self):
        _, rows = self.check([self.sub(2, "OK"), self.sub(1, "SKIPPED")])
        self.assertEqual(rows[0]["pattern"], "skips_with_later_non_skipped_attempts")
        self.assertEqual(rows[0]["replacement_compatible_skips"], 1)
        self.assertEqual(rows[0]["misconduct_label"], "unknown")

    def test_later_failure_does_not_prove_reason(self):
        _, rows = self.check([self.sub(1, "SKIPPED"), self.sub(2, "WRONG_ANSWER")])
        self.assertEqual(rows[0]["replacement_compatible_skips"], 1)
        self.assertEqual(rows[0]["misconduct_label"], "unknown")

    def test_current_out_of_competition_is_retained(self):
        _, rows = self.check([self.sub(1, "SKIPPED", "A", "OUT_OF_COMPETITION"),
                              self.sub(2, "SKIPPED", "B", "OUT_OF_COMPETITION")])
        self.assertEqual(rows[0]["pattern"], "all_skipped_multiple_problems")
        self.assertEqual(rows[0]["participation_stratum"], "OUT_OF_COMPETITION")
        self.assertEqual(rows[0]["misconduct_label"], "unknown")

    def test_single_problem_all_skipped_separate(self):
        _, rows = self.check([self.sub(1, "SKIPPED")])
        self.assertEqual(rows[0]["pattern"], "all_skipped_single_problem")

    def test_final_skipped_separate_from_replacement(self):
        _, rows = self.check([self.sub(1, "OK"), self.sub(2, "SKIPPED")])
        self.assertEqual(rows[0]["pattern"], "some_final_attempts_skipped")
        self.assertEqual(rows[0]["replacement_compatible_skips"], 0)

    def test_incomplete_and_pending_are_unknown(self):
        for items, complete in [([self.sub(1, "OK")], False),
                                ([self.sub(1, "SKIPPED"), self.sub(2, "TESTING")], True),
                                ([self.sub(1, None)], True)]:
            with self.subTest(complete=complete, items=items):
                _, rows = self.check(items, complete)
                self.assertIsNone(rows[0]["observed_skipped"])
                self.assertEqual(rows[0]["pattern"], "outcome_unresolved")

    def test_practice_virtual_and_outside_window_excluded(self):
        summary, rows = self.check([self.sub(1, "SKIPPED", kind="PRACTICE"),
                                    self.sub(2, "SKIPPED", kind="VIRTUAL"),
                                    self.sub(3, "SKIPPED", created=YEAR_END),
                                    self.sub(4, "SKIPPED", created=YEAR_START),
                                    self.sub(5, "OK")])
        self.assertEqual(rows[0]["submission_count"], 1)
        self.assertEqual(rows[0]["observed_skipped"], 0)
        self.assertEqual(sum(summary["excluded_submissions"].values()), 4)

    def test_contest_end_is_excluded(self):
        summary, rows = self.check([self.sub(1, "SKIPPED", created=YEAR_START + 7300)])
        self.assertFalse(rows)
        self.assertEqual(summary["excluded_submissions"], {"outside_contest_window": 1})

    def test_duplicate_identity_conflicts_fail(self):
        item = self.sub(1, "SKIPPED")
        summary, _ = self.check([item, item])
        self.assertEqual(summary["raw_unique_submissions"], 1)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.check([item, self.sub(1, "OK")])

    def test_invalid_ids_contest_and_missing_problem_fail(self):
        for item in [self.sub(True, "OK"), self.sub(1, "OK") | {"contestId": 2},
                     self.sub(1, "OK") | {"problem": {}}]:
            with self.assertRaises(ValueError):
                self.check([item])

    def test_mixed_participation_is_not_collapsed(self):
        _, rows = self.check([self.sub(1, "OK"), self.sub(2, "SKIPPED", kind="OUT_OF_COMPETITION")])
        self.assertEqual(rows[0]["participation_stratum"], "mixed")

    def test_before_2025_contest_rejected(self):
        with self.assertRaises(ValueError):
            audit_outcomes([], self.contest | {"startTimeSeconds": YEAR_START - 1}, True)

    def test_explicit_2026_and_default_2025_unchanged(self):
        start, end = year_bounds(2026)
        contest = self.contest | {"startTimeSeconds": start + 100}
        item = self.sub(1, "OK") | {"creationTimeSeconds": start + 101}
        summary, rows = audit_outcomes([item], contest, True, year=2026)
        self.assertEqual(rows[0]["observed_skipped"], 0)
        with self.assertRaises(ValueError):
            audit_outcomes([item], contest, True)
        summary, rows = audit_outcomes([item | {"creationTimeSeconds": end}], contest, True, year=2026)
        self.assertEqual(summary["excluded_submissions"], {"outside_2026": 1})

    def test_callback_receives_only_unique_eligible_sessions(self):
        found = []
        good = self.sub(1, "OK")
        audit_outcomes([good, good, self.sub(2, "OK", kind="PRACTICE")], self.contest, True,
                       on_session=lambda h, items: found.append((h, [s["id"] for s in items])))
        self.assertEqual(found, [("alice", [1])])

    def test_year_parameter_rejects_bool_and_invalid_year(self):
        for year in (True, 2026., 1969, 9999):
            with self.assertRaises(ValueError):
                year_bounds(year)
