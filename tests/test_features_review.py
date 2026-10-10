import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

from cf_pilot.features import (BASE_COLUMNS, FEATURE_COLUMNS, build_feature_report,
                               session_features, summarize)
from cf_pilot.review import (COLUMNS, export_review_context, export_review_register, import_reviews,
                             review_audit)
from cf_pilot.store import canonical
import test_contest_collect as fixture


def make_submission(identifier, contest, seconds, index="A", verdict="OK", language="C++"):
    value = fixture.submission(identifier, contest, seconds=seconds)
    value["problem"] = {"contestId": contest["id"], "index": index, "rating": 1200}
    value["verdict"] = verdict
    value["programmingLanguage"] = language
    return value


class PureFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.target = {"id": 10, "startTimeSeconds": 10000, "durationSeconds": 1000}
        self.items = [make_submission(1, self.target, 100, verdict="WRONG_ANSWER"),
                      make_submission(2, self.target, 110, verdict="SKIPPED"),
                      make_submission(3, self.target, 110, language="Python"),
                      make_submission(4, self.target, 200, index="B")]

    def test_hand_calculated_gaps_ties_denominators_and_diagnostics(self):
        values, diagnostic, _ = summarize(list(reversed(self.items)), self.target)
        self.assertEqual(values["attempts"], 4)
        self.assertEqual(values["attempted_problems"], 2)
        self.assertEqual(values["retries"], 2)
        self.assertEqual(values["retry_gap_min_seconds"], 0)
        self.assertEqual(values["retry_gap_p10_seconds"], 1)
        self.assertEqual(values["retry_gap_median_seconds"], 5)
        self.assertEqual(values["retry_fraction_le_15s"], 1)
        self.assertEqual(values["first_submission_fraction"], .1)
        self.assertEqual(values["attempts_per_hour"], 14.4)
        self.assertEqual(values["language_switches"], 1)
        self.assertEqual(values["problem_rating_mean"], 1200)
        self.assertEqual(diagnostic["accepted_problems_snapshot"], 2)
        self.assertEqual(diagnostic["first_accept_spacing_median_seconds"], 90)
        self.assertEqual(diagnostic["ordinary_verdict_transitions"], {})
        self.assertFalse(any("verdict" in x or "accept" in x for x in FEATURE_COLUMNS))

    def test_administrative_verdict_change_cannot_change_predictors(self):
        before = summarize(self.items, self.target)[0]
        changed = [x | {"verdict": "REJECTED"} for x in self.items]
        self.assertEqual(summarize(changed, self.target)[0], before)

    def test_single_attempt_and_missing_difficulty_are_missing_not_zero(self):
        item = make_submission(5, self.target, 300)
        item["problem"].pop("rating")
        values = summarize([item], self.target)[0]
        self.assertIsNone(values["retry_gap_median_seconds"])
        self.assertIsNone(values["retry_fraction_le_60s"])
        self.assertIsNone(values["problem_rating_mean"])
        self.assertEqual(values["problem_rating_missing_fraction"], 1)

    def test_history_minimum_zero_iqr_and_strict_future_invariance(self):
        contests = {10: self.target}
        items = list(self.items)
        for i in range(1, 6):
            contest = {"id": i, "startTimeSeconds": i * 1000, "durationSeconds": 500}
            contests[i] = contest
            items.append(make_submission(10 + i, contest, 100))
        values = session_features(items, contests, 10, range(1, 6), 180, 5)[0]
        self.assertEqual(values["prior_attempts_median"], 1)
        self.assertEqual(values["prior_attempts_iqr"], 0)
        self.assertEqual(values["prior_attempts_delta"], 3)
        self.assertIsNone(values["prior_attempts_iqr_scaled_delta"])
        self.assertIsNone(values["prior_retry_gap_median_seconds_median"])
        future = {"id": 11, "startTimeSeconds": 12000, "durationSeconds": 500}
        # An overlapping contest ends exactly at target start and is also excluded.
        overlap = {"id": 12, "startTimeSeconds": 9500, "durationSeconds": 500}
        outside = {"id": 13, "startTimeSeconds": 7000, "durationSeconds": 500}
        changed = items + [make_submission(30, future, 100), make_submission(31, overlap, 100),
                           make_submission(32, outside, 100), make_submission(33, contests[1], 500)]
        practice = make_submission(34, contests[1], 200)
        practice["author"]["participantType"] = "PRACTICE"
        changed.append(practice)
        self.assertEqual(session_features(changed, contests | {11: future, 12: overlap, 13: outside},
                                         10, [1, 2, 3, 4, 5, 11, 12], 180, 5)[0], values)
        short = session_features(items, contests, 10, [1, 2, 3, 4], 180, 4)[0]
        self.assertEqual(short["history_below_minimum"], 1)
        self.assertIsNone(short["prior_attempts_median"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            session_features(items, contests, 10, [1, 2], 180, 5)

    def test_iqr_calculation_with_variable_history(self):
        contests = {10: self.target}
        items = list(self.items)
        for i in range(1, 6):
            contest = {"id": i, "startTimeSeconds": i * 1000, "durationSeconds": 500}
            contests[i] = contest
            items.extend(make_submission(100 * i + j, contest, 10 * j) for j in range(i))
        values = session_features(items, contests, 10, range(1, 6), 180)[0]
        self.assertEqual(values["prior_attempts_median"], 3)
        self.assertEqual(values["prior_attempts_iqr"], 2)
        self.assertEqual(values["prior_attempts_iqr_scaled_delta"], .5)

    def test_duplicates_and_conflicting_difficulties_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(self.items + [self.items[0]], self.target)
        changed = [dict(x) for x in self.items]
        changed[1] = changed[1] | {"problem": changed[1]["problem"] | {"rating": 1800}}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            summarize(changed, self.target)


class ReportAndReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ContestTests("test_end_to_end_population_history_sample_and_offline_rerun")
        self.fixture.setUp()
        self.fixture.run_fixture()
        self.store = self.fixture.store
        self.output = Path(self.fixture.temp.name) / "features"
        self.review_path = Path(self.fixture.temp.name) / "review.csv"

    def tearDown(self):
        self.fixture.tearDown()

    def rows(self):
        with self.review_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def write_rows(self, rows):
        with self.review_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def positive(self, row):
        reviewed = datetime.fromtimestamp(self.fixture.target["startTimeSeconds"] + 9000, timezone.utc).isoformat()
        collected = datetime.fromtimestamp(self.fixture.target["startTimeSeconds"] + 8000, timezone.utc).isoformat()
        return row | {"label": "substantiated_misconduct", "evidence_url": "https://example.org/finding",
                      "evidence_type": "platform_finding", "source_collected_at": collected,
                      "reviewed_at": reviewed, "reviewer": "reviewer_one",
                      "confidence": "high", "session_verified": "yes", "related_case_id": "case_001",
                      "reviewer_notes": "Fixture source, not a real allegation."}

    def export(self):
        return export_review_register(self.store, "test", self.review_path, limit=100)

    def test_feature_pipeline_deterministic_allowlist_no_labels_and_frozen_files(self):
        manifest = build_feature_report(self.store, "test", self.output)
        self.assertEqual(manifest["rows"], 3)
        self.assertEqual(manifest["model_columns"], list(FEATURE_COLUMNS))
        self.assertEqual(len(set(FEATURE_COLUMNS)), len(FEATURE_COLUMNS))
        self.assertFalse(set(FEATURE_COLUMNS) & {"handle", "college", "country", "in_standings", "label"})
        before = {x.name: x.read_bytes() for x in self.output.iterdir()}
        # Neither changed outcome labels nor standings can alter predictors or the input hash.
        self.store.db.execute("UPDATE labels SET label='unknown',reviewer_notes='new note'")
        self.store.db.execute("UPDATE standings SET payload_json='{}'")
        self.store.db.commit()
        self.assertEqual(build_feature_report(self.store, "test", self.output), manifest)
        self.assertEqual({x.name: x.read_bytes() for x in self.output.iterdir()}, before)
        for name, digest in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
        (self.output / "features.csv").write_text("manual change")
        with self.assertRaisesRegex(ValueError, "Frozen"):
            build_feature_report(self.store, "test", self.output)
        self.assertEqual((self.output / "features.csv").read_text(), "manual change")

    def test_incomplete_collection_rejected(self):
        self.store.db.execute("UPDATE jobs SET status='incomplete' WHERE job_key='contest:1'")
        self.store.db.commit()
        with self.assertRaisesRegex(ValueError, "audit"):
            build_feature_report(self.store, "test", self.output)
        self.assertFalse(self.output.exists())

    def test_export_preserves_manual_work_and_unreviewed_import_noop(self):
        self.assertEqual(self.export()["rows"], 3)
        template = self.review_path.read_bytes()
        before = list(self.store.db.execute("SELECT * FROM labels"))
        result = import_reviews(self.store, "test", self.review_path, apply=True)
        self.assertEqual(result["new_review_events"], 0)
        self.assertEqual([tuple(x) for x in self.store.db.execute("SELECT * FROM labels")], [tuple(x) for x in before])
        self.review_path.write_text("keep handwritten notes")
        self.assertTrue(self.export()["preserved_existing"])
        self.assertEqual(self.review_path.read_text(), "keep handwritten notes")
        self.review_path.write_bytes(template)
        other = Path(self.fixture.temp.name) / "other.csv"
        export_review_register(self.store, "test", other)
        self.assertEqual(other.read_bytes(), template)

    def test_atomic_validation_rejects_one_bad_row_without_applying_good_row(self):
        self.export()
        rows = self.rows()
        rows[0] = self.positive(rows[0])
        rows[1] = self.positive(rows[1]) | {"evidence_type": "fast_retries"}
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "independent"):
            import_reviews(self.store, "test", self.review_path, apply=True)
        self.assertEqual(review_audit(self.store, "test")["eligible_session_labels"], {"unknown": 4})
        self.assertEqual(review_audit(self.store, "test")["review_events"], 0)

    def test_dry_run_apply_idempotency_and_reversal_history(self):
        self.export()
        rows = self.rows()
        rows[0] = self.positive(rows[0])
        self.write_rows(rows)
        self.assertFalse(import_reviews(self.store, "test", self.review_path)["applied"])
        self.assertEqual(review_audit(self.store, "test")["review_events"], 0)
        self.assertEqual(import_reviews(self.store, "test", self.review_path, True)["new_review_events"], 1)
        self.assertEqual(import_reviews(self.store, "test", self.review_path, True)["new_review_events"], 0)
        rows[0] = rows[0] | {"label": "unknown", "reversal": "yes", "reviewer_notes": "Finding reversed."}
        self.write_rows(rows)
        self.assertEqual(import_reviews(self.store, "test", self.review_path, True)["new_review_events"], 1)
        self.assertEqual(review_audit(self.store, "test")["eligible_session_labels"], {"unknown": 4})
        self.assertEqual(review_audit(self.store, "test")["review_events"], 2)

    def test_ai_disagreement_and_session_identity_validation(self):
        self.export()
        base = self.positive(self.rows()[0])
        for changes in [{"label": "substantiated_ai_misuse"}, {"disagreement": "yes"},
                        {"reversal": "yes"}, {"evidence_url": "file:///finding"},
                        {"reviewed_at": "2020-01-02"}, {"source_collected_at": "2020-01-03T00:00:00Z"},
                        {"handle": "practice_only"}, {"submission_ids": "99999"},
                        {"second_reviewer": "reviewer_two", "second_decision": "unknown"},
                        {"second_reviewer": "reviewer_one", "second_decision": "substantiated_misconduct"}]:
            self.write_rows([base | changes])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                import_reviews(self.store, "test", self.review_path, True)
        self.write_rows([base | {"label": "substantiated_ai_misuse", "ai_explicit": "yes"}])
        self.assertEqual(import_reviews(self.store, "test", self.review_path)["reviewed_rows"], 1)

    def test_presumed_negative_requires_review_and_unknown_never_forced_negative(self):
        self.export()
        row = self.positive(self.rows()[0]) | {"label": "presumed_negative", "evidence_url": "",
               "evidence_type": "reviewed_no_independent_finding", "related_case_id": ""}
        self.write_rows([row])
        self.assertEqual(import_reviews(self.store, "test", self.review_path, True)["labels"], {"presumed_negative": 1})
        counts = review_audit(self.store, "test")["eligible_session_labels"]
        self.assertEqual(counts["unknown"], 3)
        self.assertEqual(counts["presumed_negative"], 1)
        self.assertFalse(review_audit(self.store, "test")["training_ready"])

    def test_duplicate_session_and_header_rejected(self):
        self.export()
        row = self.rows()[0]
        self.write_rows([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate session"):
            import_reviews(self.store, "test", self.review_path)
        self.review_path.write_text("run_id,handle,label\ntest,alice,fair\n")
        with self.assertRaisesRegex(ValueError, "header"):
            import_reviews(self.store, "test", self.review_path)

    def test_review_audit_detects_direct_labels_without_evidence(self):
        self.export()
        row = self.positive(self.rows()[0])
        self.write_rows([row])
        import_reviews(self.store, "test", self.review_path, True)
        self.assertEqual(review_audit(self.store, "test")["issues"], [])
        self.store.db.execute("UPDATE labels SET reviewer_notes='untracked edit' WHERE handle=?", (row["handle"],))
        self.store.db.commit()
        self.assertEqual(review_audit(self.store, "test")["issues"][0]["code"], "label_without_matching_review_event")

    def test_submission_reference_must_match_handle_and_contest(self):
        self.export()
        row = self.positive(self.rows()[0])
        correct_id = self.store.db.execute("SELECT submission_id FROM submissions WHERE run_id='test' "
            "AND contest_id=3 AND handle=?", (row["handle"],)).fetchone()[0]
        self.write_rows([row | {"submission_ids": str(correct_id)}])
        self.assertEqual(import_reviews(self.store, "test", self.review_path)["reviewed_rows"], 1)
        wrong_id = self.store.db.execute("SELECT submission_id FROM submissions WHERE run_id='test' "
            "AND contest_id=3 AND handle!=? LIMIT 1", (row["handle"],)).fetchone()[0]
        self.write_rows([row | {"submission_ids": str(wrong_id)}])
        with self.assertRaisesRegex(ValueError, "belong"):
            import_reviews(self.store, "test", self.review_path)

    def test_review_context_matches_official_records_without_scores(self):
        self.export()
        output = self.review_path.with_suffix(".context.json")
        result = export_review_context(self.store, "test", self.review_path, output)
        self.assertEqual(result["cases"], 3)
        data = json.loads(output.read_text())
        for case in data["cases"]:
            self.assertEqual(case["official_submission_count"], 1)
            self.assertEqual(case["submissions"][0]["offset_seconds"], 30)
            self.assertNotIn("label", case)
            self.assertNotIn("score", case)
        before = output.read_bytes()
        export_review_context(self.store, "test", self.review_path, output)
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
