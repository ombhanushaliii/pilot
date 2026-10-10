import csv
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

from cf_pilot.api import Client, CollectionError
from cf_pilot.audit import audit_run, write_reports
from cf_pilot.collect import collect_pages, load_config
from cf_pilot.contest_collect import (DEFAULTS, build_contest_sessions, choose_panel,
                                     collect_contest, stratified_sample, validate_config)
from cf_pilot.store import Store, canonical


def submission(identifier, contest, handle="alice", kind="CONTESTANT", seconds=30):
    return {"id": identifier, "contestId": contest["id"],
            "creationTimeSeconds": contest["startTimeSeconds"] + seconds,
            "relativeTimeSeconds": seconds, "problem": {"contestId": contest["id"], "index": "A"},
            "author": {"participantType": kind, "members": [{"handle": handle}]},
            "programmingLanguage": "GNU C++17", "verdict": "OK"}


def response(items):
    return json.dumps({"status": "OK", "result": items}).encode()


class ContestTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "work" / "test_tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.store = Store(Path(self.temp.name))
        self.config = validate_config(DEFAULTS | {"run_id": "test", "history_contest_count": 2,
                                                   "regular_min_contests": 2, "sample_size": 3, "page_size": 3})
        self.store.start_run("test", self.config)
        self.now = self.store.db.execute("SELECT created_at FROM runs WHERE run_id='test'").fetchone()[0]
        self.target = {"id": 3, "name": "Round (Div. 2)", "type": "CF", "phase": "FINISHED",
                       "startTimeSeconds": int(self.now) - 10 * 86400, "durationSeconds": 7200}
        self.earlier = [self.target | {"id": 2, "startTimeSeconds": int(self.now) - 20 * 86400},
                        self.target | {"id": 1, "startTimeSeconds": int(self.now) - 30 * 86400}]
        self.contests = [self.target] + self.earlier
        self.time = 100.0

    def tearDown(self):
        self.store.close()
        if not Path(self.temp.name).resolve().is_relative_to(self.temp_root.resolve()):
            raise AssertionError("Cleanup escaped project work directory")
        self.temp.cleanup()

    def sleep(self, seconds):
        self.time += seconds

    def client(self, transport):
        return Client(self.store, "test", transport=transport, clock=lambda: self.time, sleep=self.sleep)

    def fixture_transport(self, calls):
        # Alice is regular, Bob limited, Carol and Dave have no observed history.
        # One eligible participant is absent from standings and must be retained.
        self.records = {3: [submission(305, self.target, "alice"), submission(304, self.target, "bob"),
                            submission(303, self.target, "carol"), submission(302, self.target, "dave"),
                            submission(301, self.target, "practice_only", "PRACTICE")],
                        2: [submission(205, self.earlier[0], "alice"), submission(204, self.earlier[0], "bob"),
                            submission(203, self.earlier[0], "carol", "PRACTICE"),
                            submission(202, self.earlier[0], "outsider")],
                        1: [submission(105, self.earlier[1], "alice"), submission(104, self.earlier[1], "bob", seconds=7200)]}
        def transport(url, timeout):
            method = urlsplit(url).path.split("/")[-1]
            params = parse_qs(urlsplit(url).query)
            calls.append((method, params))
            if method == "contest.list":
                return response(self.contests)
            if method == "contest.standings":
                self.assertEqual(set(params), {"contestId"})
                return response({"contest": self.target, "rows": [{"party": {"members": [{"handle": h}]}} for h in ["alice", "bob", "carol"]]})
            if method == "contest.status":
                self.assertNotIn("handle", params)
                identifier = int(params["contestId"][0])
                index = int(params["from"][0]) - 1
                count = int(params["count"][0])
                return response(self.records[identifier][index:index + count])
            raise AssertionError(url)
        return transport

    def run_fixture(self):
        calls = []
        client = self.client(self.fixture_transport(calls))
        self.assertTrue(collect_contest(client, self.config, progress=lambda text: None))
        self.store.db.execute("UPDATE runs SET status='complete' WHERE run_id='test'")
        self.store.db.commit()
        return client, calls

    def test_config_dispatch_and_legacy_config_unchanged(self):
        path = Path(self.temp.name) / "contest.json"
        path.write_text(json.dumps({"sampling_strategy": "contest"}))
        self.assertEqual(load_config(path), DEFAULTS)
        legacy_path = Path(__file__).resolve().parents[1] / "config" / "pilot.json"
        legacy = load_config(legacy_path)
        self.assertNotIn("sampling_strategy", legacy)
        for changes in [{"regular_min_contests": 21}, {"sample_weights": {"regular": 1}},
                        {"sample_weights": {"regular": float("nan"), "limited": 1, "none_observed": 1}},
                        {"target_contest_id": -1}, {"organization_url": "unused"}]:
            with self.assertRaises(ValueError):
                validate_config(DEFAULTS | changes)

    def test_selection_is_temporal_deterministic_and_mature(self):
        future = self.target | {"id": 5, "startTimeSeconds": int(self.now) + 100}
        young = self.target | {"id": 4, "startTimeSeconds": int(self.now) - 86400}
        wrong_type = self.earlier[0] | {"id": 6, "type": "ICPC"}
        target, history = choose_panel(list(reversed(self.contests + [future, young, wrong_type])), self.config, self.now)
        self.assertEqual(target["id"], 3)
        self.assertEqual([x["id"] for x in history], [2, 1])
        with self.assertRaises(CollectionError):
            choose_panel(self.contests, self.config | {"target_contest_id": 4}, self.now)
        with self.assertRaises(CollectionError):
            choose_panel(self.contests[:2], self.config, self.now)

    def test_overlapping_and_combined_history_excluded_but_duration_can_vary(self):
        overlap = self.target | {"id": 7, "startTimeSeconds": self.target["startTimeSeconds"] - 100}
        long_round = self.earlier[0] | {"id": 8, "durationSeconds": 10800}
        combined = self.earlier[0] | {"id": 9, "name": "Round (Div. 1 + Div. 2)"}
        unrated = self.earlier[0] | {"id": 10, "name": "Round (Unrated, Div. 2)"}
        _, panel = choose_panel(self.contests + [overlap, long_round, combined, unrated], self.config, self.now)
        self.assertEqual([x["id"] for x in panel], [8, 2])

    def test_sampling_redistributes_shortages_and_is_order_independent(self):
        groups = {"regular": [f"r{x}" for x in range(2)], "limited": [f"l{x}" for x in range(20)], "none_observed": []}
        selected, probabilities = stratified_sample(groups, 10, 736, DEFAULTS["sample_weights"])
        self.assertEqual(len(selected), 10)
        self.assertTrue({"r0", "r1"} <= selected)
        self.assertEqual(probabilities["regular"], 1)
        self.assertEqual(probabilities["limited"], 0.4)
        reversed_groups = {key: list(reversed(values)) for key, values in reversed(list(groups.items()))}
        self.assertEqual((selected, probabilities), stratified_sample(reversed_groups, 10, 736, DEFAULTS["sample_weights"]))
        all_selected, _ = stratified_sample(groups, 1000, 736, DEFAULTS["sample_weights"])
        self.assertEqual(len(all_selected), 22)

    def test_short_pages_do_not_silently_truncate_census(self):
        items = [submission(10 - x, self.target) for x in range(5)]
        indices = []
        def clipped(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            indices.append(index)
            # Emulate server returning fewer than the requested count.
            return response(items[index - 1:index + 1])
        complete = collect_pages(self.client(clipped), "contest:3", "contest.status", {"contestId": 3}, 10, 10, exhaust_on_empty=True)
        self.assertTrue(complete)
        self.assertEqual(indices, [1, 2, 3, 4, 5, 6])
        self.assertEqual(json.loads(self.store.job("test", "contest:3")["detail"])["unique_items"], 5)

    def test_shifted_page_fails_instead_of_dropping_records(self):
        pages = {1: [9, 8, 7], 3: [6, 5, 4]}
        def shifted(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            return response([submission(x, self.target) for x in pages[index]])
        with self.assertRaisesRegex(CollectionError, "overlap changed"):
            collect_pages(self.client(shifted), "contest:3", "contest.status", {"contestId": 3}, 3, 10, exhaust_on_empty=True)

    def test_front_insertions_realign_without_including_new_records_or_losing_old(self):
        pages = {1: [9, 8, 7], 3: [8, 7, 6], 5: [6, 5], 6: [5], 7: []}
        indices = []
        def inserted(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            indices.append(index)
            return response([submission(x, self.target) for x in pages[index]])
        self.assertTrue(collect_pages(self.client(inserted), "contest:3", "contest.status", {"contestId": 3}, 3, 10, exhaust_on_empty=True))
        identifiers = {row[0] for row in self.store.db.execute("SELECT submission_id FROM submissions")}
        self.assertEqual(identifiers, {9, 8, 7, 6, 5})
        self.assertEqual(indices, [1, 3, 5, 6, 7])
        detail = json.loads(self.store.job("test", "contest:3")["detail"])
        self.assertEqual(detail["unique_items"], 5)
        self.assertEqual(detail["overlap_realignments"], 1)
        self.assertEqual(detail["highest_id"], 9)

    def test_contest_page_checkpoint_and_participation_sink_resume(self):
        calls = []
        transport = self.fixture_transport(calls)
        def interrupted(url, timeout):
            params = parse_qs(urlsplit(url).query)
            if '/contest.status' in url and params['contestId']==['2'] and int(params['from'][0]) > 1:
                raise RuntimeError('Simulated interruption in historical paging')
            return transport(url, timeout)
        with self.assertRaises(CollectionError):
            collect_contest(self.client(interrupted), self.config, progress=lambda text: None)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM participant_coverage").fetchone()[0], 0)
        self.assertEqual(self.store.job('test','contest:2')['next_from'], 3)
        detail=json.loads(self.store.job('test','contest:2')['detail'])
        detail.pop('highest_id')
        self.store.db.execute("UPDATE jobs SET detail=? WHERE run_id='test' AND job_key='contest:2'",(canonical(detail),))
        self.store.db.commit()
        calls.clear()
        self.assertTrue(collect_contest(self.client(transport), self.config, progress=lambda text: None))
        self.assertFalse(any(method=='contest.status' and params['contestId']==['2'] and params['from']==['1'] for method,params in calls))
        self.store.db.execute("UPDATE runs SET status='complete' WHERE run_id='test'")
        self.store.db.commit()
        report=audit_run(self.store,'test')
        self.assertTrue(report['collection_checks_passed'],report['issues'])
        self.assertEqual(report['population']['coverage_groups']['regular'],1)
        self.assertEqual(json.loads(self.store.job('test','contest:2')['detail'])['highest_id'],205)

    def test_end_to_end_population_history_sample_and_offline_rerun(self):
        client, calls = self.run_fixture()
        report = audit_run(self.store, "test")
        self.assertTrue(report["collection_checks_passed"], report["issues"])
        self.assertEqual(report["counts"]["eligible_sessions"], 4)
        self.assertEqual(report["counts"]["sessions"], 5)
        self.assertEqual(report["population"]["coverage_groups"], {"regular": 1, "limited": 1, "none_observed": 2})
        self.assertEqual(report["population"]["development_sample_size"], 3)
        self.assertEqual(report["counts"]["labels"], {"unknown": 5})
        self.assertEqual(len(report["submission_sessions_absent_from_standings"]), 1)
        self.assertTrue(all(method.startswith("contest.") for method, _ in calls))
        before = report["counts"]
        count = client.network_requests
        self.assertTrue(collect_contest(client, self.config, progress=lambda text: None))
        self.assertEqual(client.network_requests, count)
        self.assertEqual(audit_run(self.store, "test")["counts"], before)
        output = Path(self.temp.name) / "reports"
        write_reports(self.store, "test", output, report)
        with (output / "development_sample.csv").open(newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)
        with (output / "participant_coverage.csv").open(newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 4)
        notes = output / "labels_for_review.csv"
        notes.write_text("keep manual notes")
        write_reports(self.store, "test", output, report)
        self.assertEqual(notes.read_text(), "keep manual notes")

    def test_future_and_practice_records_cannot_change_coverage(self):
        self.run_fixture()
        before = [tuple(row) for row in self.store.db.execute("SELECT * FROM participant_coverage ORDER BY handle")]
        future = self.target | {"id": 4, "startTimeSeconds": int(self.now) + 100}
        self.store.db.execute("INSERT INTO contests VALUES(?,?,?,0)", ("test", 4, canonical(future)))
        self.store.db.execute("INSERT INTO panel_participation VALUES(?,?,?)", ("test", 4, "carol"))
        self.store.db.commit()
        build_contest_sessions(self.store, "test", self.config)
        self.assertEqual([tuple(row) for row in self.store.db.execute("SELECT * FROM participant_coverage ORDER BY handle")], before)

    def test_audit_rejects_missing_sample_history_and_bad_sampling_probability(self):
        self.run_fixture()
        self.store.db.execute("DELETE FROM submissions WHERE run_id='test' AND contest_id=1")
        self.store.db.execute("UPDATE participant_coverage SET selection_probability=0")
        self.store.db.commit()
        codes = {x["code"] for x in audit_run(self.store, "test")["issues"]}
        self.assertIn("sample_history_normalization_incomplete", codes)
        self.assertIn("sampling_probability_mismatch", codes)

    def test_incomplete_panel_is_not_classified_as_no_history(self):
        calls = []
        complete = collect_contest(self.client(self.fixture_transport(calls)), self.config | {"max_contest_pages": 1}, progress=lambda text: None)
        self.assertFalse(complete)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM participant_coverage").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT sum(history_complete) FROM sessions").fetchone()[0], 0)
        self.assertFalse(audit_run(self.store, "test")["collection_checks_passed"])


if __name__ == "__main__":
    unittest.main()
