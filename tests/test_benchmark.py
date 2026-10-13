import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cf_pilot.api import request_key
from cf_pilot.benchmark import (FEATURE_COLUMNS, behavior_features, build_benchmark,
    file_hash, fit_preprocessor, recorded_target, replay_archives, source_manifest,
    transform_features, validate_partitions)
from cf_pilot.outcomes import year_bounds
from cf_pilot.store import Store, canonical


START = year_bounds(2026)[0]


def contest(cid=1, offset=100):
    return {"id": cid, "name": "Fixture Div. 2", "startTimeSeconds": START + offset,
            "durationSeconds": 3600}


def submission(sid, problem="A", seconds=10, verdict="OK", handle="Alice", kind="CONTESTANT", c=None):
    c = c or contest()
    return {"id": sid, "contestId": c["id"], "creationTimeSeconds": c["startTimeSeconds"] + seconds,
            "problem": {"index": problem}, "programmingLanguage": "C++17", "verdict": verdict,
            "author": {"members": [{"handle": handle}], "participantType": kind}}


class BenchmarkFeatureTests(unittest.TestCase):
    def test_features_ignore_labels_authors_and_judging_fields(self):
        items = [submission(1), submission(2, "B", 20), submission(3, "A", 30)]
        original = behavior_features(items, contest())
        changed = copy.deepcopy(items)
        for s in changed:
            s.update(verdict="SKIPPED", passedTestCount=0, timeConsumedMillis=0, memoryConsumedBytes=0,
                     points=0, testset="CHALLENGES", relativeTimeSeconds=-1)
            s["author"] = {"participantType": "OUT_OF_COMPETITION", "members": [{"handle": "Elsewhere"}], "organization": "ignored"}
            s["problem"].update(rating=3500, tags=["ignored"])
        self.assertEqual(original, behavior_features(changed, contest()))
        self.assertEqual(set(original), set(FEATURE_COLUMNS))
        self.assertEqual(original["retries"], 1)
        self.assertEqual(original["problem_revisit_fraction"], .5)

    def test_sorting_problem_rename_and_timestamp_ties(self):
        items = [submission(2, "B", 10), submission(1, "A", 10)]
        self.assertEqual(behavior_features(items, contest()), behavior_features(items[::-1], contest()))
        renamed = copy.deepcopy(items)
        for s in renamed:
            s["problem"]["index"] += "ignored-name"
        self.assertEqual(behavior_features(items, contest()), behavior_features(renamed, contest()))
        self.assertEqual(behavior_features(items, contest())["gap_min_fraction"], 0)

    def test_missing_languages_and_retry_gaps_remain_missing(self):
        items = [submission(1), submission(2, "B", 20)]
        for s in items:
            s.pop("programmingLanguage")
        row = behavior_features(items, contest())
        self.assertIsNone(row["retry_gap_min_fraction"])
        self.assertIsNone(row["language_switch_fraction"])
        self.assertEqual(row["language_missing_fraction"], 1)

    def test_duplicate_outside_contest_and_empty_fail(self):
        item = submission(1)
        for items in ([], [item, item], [item | {"creationTimeSeconds": START}]):
            with self.assertRaises(ValueError):
                behavior_features(items, contest())

    def test_target_definition_and_exclusions(self):
        def row(pattern, n=2):
            return {"pattern": pattern, "problem_count": n}
        self.assertEqual(recorded_target(row("all_skipped_multiple_problems")), (1, ""))
        self.assertEqual(recorded_target(row("no_skipped_observed")), (0, ""))
        for pattern in ("some_final_attempts_skipped", "skips_with_later_non_skipped_attempts"):
            self.assertEqual(recorded_target(row(pattern)), (None, "ambiguous_skipped_pattern"))
        self.assertEqual(recorded_target(row("no_skipped_observed", 1)), (None, "single_problem"))
        self.assertEqual(recorded_target(row("outcome_unresolved", 1)), (None, "outcome_unresolved"))

    def test_training_preprocessor_masks_and_holdout_invariance(self):
        a = behavior_features([submission(1), submission(2, "B", 20)], contest())
        b = a | {"attempts": 4}
        pre = fit_preprocessor([a, b])
        before = canonical(pre)
        vector = transform_features(a | {"attempts": 1000000}, pre)
        self.assertEqual(canonical(pre), before)
        self.assertEqual(pre["stats"]["attempts"]["mean"], 3)
        self.assertEqual(len(vector), 2 * len(FEATURE_COLUMNS))
        idx = FEATURE_COLUMNS.index("retry_gap_min_fraction")
        self.assertEqual(vector[len(FEATURE_COLUMNS) + idx], 1)

    def test_extra_target_and_nonfinite_inputs_rejected(self):
        row = behavior_features([submission(1)], contest())
        for bad in (row | {"recorded_outcome": 1}, row | {"attempts": float("nan")}, row | {"attempts": True}):
            with self.assertRaises(ValueError):
                fit_preprocessor([bad])
        with self.assertRaises(ValueError):
            fit_preprocessor([])

    def test_contest_overlap_temporal_overlap_and_wrong_year_rejected(self):
        cs = {cid: contest(cid, cid * 10000) for cid in (1, 2, 3)}
        config = {"year": 2026, "splits": {"train": [1], "validation": [2], "test": [3]}}
        validate_partitions(config, cs)
        for bad in (config | {"year": 2025}, config | {"splits": {"train": [1], "validation": [1, 2], "test": [3]}},
                    config | {"splits": {"train": [3], "validation": [2], "test": [1]}}):
            with self.assertRaises(ValueError):
                validate_partitions(bad, cs)


class BenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1] / "work"
        workspace.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=workspace)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data"
        self.store = Store(self.source)
        self.store.start_run("source", {})
        self.config = {"benchmark_id": "fixture", "source_data": "data", "source_run": "source",
                       "year": 2026, "splits": {"train": [1], "validation": [2], "test": [3]}}
        self.pages = {}
        for cid in (1, 2, 3):
            c = contest(cid, cid * 10000)
            self.store.db.execute("INSERT INTO contests VALUES(?,?,?,1)", ("source", cid, canonical(c)))
            self.store.db.execute("INSERT INTO contest_panel VALUES(?,?,?,?)", ("source", cid, "target" if cid == 3 else "history", cid))
            items = [submission(cid * 100 + 1, "A", 10, "SKIPPED", "bad", "OUT_OF_COMPETITION", c),
                     submission(cid * 100 + 2, "B", 20, "SKIPPED", "bad", "OUT_OF_COMPETITION", c),
                     submission(cid * 100 + 3, "A", 30, "OK", "good", c=c),
                     submission(cid * 100 + 4, "B", 40, "WRONG_ANSWER", "good", c=c),
                     submission(cid * 100 + 5, "A", 50, "OK", "single", c=c)]
            items.sort(key=lambda s: s["id"], reverse=True)
            for offset, values in ((1, items), (5, [items[-1]]), (6, [])):
                params = {"contestId": cid, "from": offset, "count": 5}
                url = "https://codeforces.com/api/contest.status"
                key = request_key(url, params)
                body = canonical({"status": "OK", "result": values}).encode()
                self.store.archive("source", key, url, params, body)
                self.pages[(cid, offset)] = key
            self.store.job("source", f"contest:{cid}")
            self.store.update_job("source", f"contest:{cid}", 6, "complete", 3,
                canonical({"highest_id": items[0]["id"], "unique_items": 5, "stop_reason": "exhausted"}))
        self.store.db.execute("UPDATE runs SET status='complete'")
        self.store.db.commit()
        self.store.close()

    def quiet(self, *args, **kwargs):
        pass

    def test_offline_full_build_resume_and_source_immutability(self):
        path = self.source / "pilot.sqlite"
        before = file_hash(path)
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be used")):
            result = build_benchmark(self.root, self.config, self.quiet)
            self.assertTrue(result["complete"])
            self.assertEqual(result["partitions"]["test"]["class_counts"], {"positive": 1, "comparison": 1})
            destination = self.root / "data/fixture"
            snapshot = {str(p.relative_to(destination)): file_hash(p) for p in destination.rglob("*.csv")}
            self.assertEqual(result, build_benchmark(self.root, self.config, self.quiet))
            self.assertEqual(snapshot, {str(p.relative_to(destination)): file_hash(p) for p in destination.rglob("*.csv")})
            self.assertEqual(file_hash(path), before)
            self.assertEqual(result["partitions"]["user_overlap"]["train:test"], 2)
            self.assertEqual(result["contests"][0]["benchmark_counts"]["excluded:single_problem"], 1)
            with (destination / "train/features.csv").open() as s:
                self.assertEqual(csv.DictReader(s).fieldnames, ["session_id"] + list(FEATURE_COLUMNS))
            pre = json.loads((destination / "preprocessor.json").read_text())
            self.assertEqual(pre["rows"], 2)
            self.assertEqual(pre["fitted_on"], "train")

    def test_corrupt_source_archive_fails(self):
        db = __import__("sqlite3").connect(self.source / "pilot.sqlite")
        path = db.execute("SELECT raw_path FROM responses WHERE request_key=?", (self.pages[(1, 1)],)).fetchone()[0]
        db.close()
        (self.source / path).write_bytes(b"not gzip")
        with self.assertRaises((OSError, ValueError)):
            build_benchmark(self.root, self.config, self.quiet)

    def test_incomplete_source_census_fails_before_build(self):
        db = __import__("sqlite3").connect(self.source / "pilot.sqlite")
        db.execute("UPDATE jobs SET status='failed' WHERE job_key='contest:1'")
        db.commit()
        db.close()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            build_benchmark(self.root, self.config, self.quiet)

    def test_legacy_checkpoint_recovers_highest_from_verified_first_page(self):
        db = __import__("sqlite3").connect(self.source / "pilot.sqlite")
        detail = json.loads(db.execute("SELECT detail FROM jobs WHERE job_key='contest:1'").fetchone()[0])
        detail.pop("highest_id")
        db.execute("UPDATE jobs SET detail=? WHERE job_key='contest:1'", (canonical(detail),))
        db.commit()
        db.close()
        result = build_benchmark(self.root, self.config, self.quiet)
        self.assertTrue(result["complete"])
        self.assertEqual(result["contests"][0]["audit"]["raw_unique_submissions"], 5)

    def test_snapshot_boundary_mismatch_fails(self):
        db = __import__("sqlite3").connect(self.source / "pilot.sqlite")
        detail = json.loads(db.execute("SELECT detail FROM jobs WHERE job_key='contest:1'").fetchone()[0])
        detail["highest_id"] += 1
        db.execute("UPDATE jobs SET detail=? WHERE job_key='contest:1'", (canonical(detail),))
        db.commit()
        db.close()
        with self.assertRaisesRegex(ValueError, "highest ID"):
            build_benchmark(self.root, self.config, self.quiet)

    def test_modified_completed_output_and_changed_config_rejected(self):
        build_benchmark(self.root, self.config, self.quiet)
        with self.assertRaisesRegex(ValueError, "Frozen benchmark"):
            build_benchmark(self.root, self.config | {"extra": "different"}, self.quiet)
        p = self.root / "data/fixture/contests/1/features.csv"
        p.write_text(p.read_text() + "tampered\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            build_benchmark(self.root, self.config, self.quiet)

    def test_archive_path_cannot_escape_source_root(self):
        from collections import Counter
        source = {"job_detail": {"highest_id": 10}, "pages": [{"raw_path": "../elsewhere"}]}
        with self.assertRaisesRegex(ValueError, "outside source"):
            list(replay_archives(self.source, source, Counter()))
