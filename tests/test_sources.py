import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cf_pilot.sources import (Archive, PROJECT, SourceError, collect_sources,
                              exclusive_file, read_config, source_url, validate_snapshot)


def task(identifier=100):
    return {"contest_id": 2211, "submission_id": identifier,
            "expected_handle": "Example", "purpose": "unverified_evidence_lead"}


def config():
    return {"version": 1, "run_id": "source-test-001", "submissions": [task(), task(101)],
            "request_interval_seconds": 7, "timeout_seconds": 30,
            "min_free_mb": 1024, "max_process_rss_mb": 1000}


def snapshot(item):
    return {"title": "Submission - Codeforces", "challenge": False,
            "source_count": 1, "source": "  // λ\nint main() {\n\treturn 0;\n}\n\n",
            "rows": [{"submission_href": source_url(item), "authors": ["Example"],
                      "language": "GNU C++17", "problem_href": "/contest/2211/problem/F"}]}


class Memory:
    def check(self):
        pass


class SourceTests(unittest.TestCase):
    def setUp(self):
        (PROJECT / "work").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "work")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_validation_and_exact_source_whitespace(self):
        item = task()
        value = snapshot(item)
        text, identity = validate_snapshot(item, source_url(item), 200, value)
        self.assertEqual(text, value["source"])
        self.assertEqual(identity["problem_index"], "F")
        self.assertEqual(identity["observed_handle"], "Example")
        self.assertEqual(identity["language"], "GNU C++17")
        value["rows"][0]["authors"] = ["example"]
        validate_snapshot(item, source_url(item), 200, value)

    def test_denied_redirect_empty_duplicate_and_wrong_owner_fail(self):
        item = task()
        for url, status in [(source_url(item), 403), (source_url(item), 429),
                            ("https://codeforces.com/enter", 200),
                            ("https://elsewhere.example/contest/2211/submission/100", 200)]:
            with self.subTest(url=url, status=status), self.assertRaises(SourceError):
                validate_snapshot(item, url, status, snapshot(item))
        for replacement in [{"source": "  "}, {"source_count": 2}, {"challenge": True}, {"rows": []}]:
            with self.subTest(replacement=replacement), self.assertRaises(SourceError):
                validate_snapshot(item, source_url(item), 200, snapshot(item) | replacement)
        for key, value in [("authors", ["Wrong"]), ("authors", ["Example", "Teammate"]),
                           ("problem_href", "/contest/2269/problem/F"), ("language", "")]:
            bad = snapshot(item)
            bad["rows"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(SourceError):
                validate_snapshot(item, source_url(item), 200, bad)

    def test_archive_checksum_frozen_manifest_and_no_overwrite(self):
        archive = Archive(self.root, config())
        item = task()
        text, identity = validate_snapshot(item, source_url(item), 200, snapshot(item))
        archive.save(item, text, identity)
        source, metadata = archive.paths(item)
        self.assertEqual(source.read_bytes(), text.encode("utf-8"))
        self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["label"], "unknown")
        self.assertTrue(archive.cached(item))
        with self.assertRaises(SourceError):
            archive.save(item, text, identity)
        with self.assertRaises(ValueError):
            Archive(self.root, config() | {"request_interval_seconds": 8})
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(SourceError, "checksum"):
            archive.cached(item)

    def test_denial_stops_and_resume_skips_verified_successes(self):
        archive = Archive(self.root, config())
        calls = []
        def fetch(item):
            calls.append(item["submission_id"])
            if item["submission_id"] == 101:
                raise SourceError("access_challenge_or_denial")
            return source_url(item), 200, snapshot(item)
        first = collect_sources(config(), archive, fetch, Memory(), 5, sleep=lambda _: None)
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(first["downloaded"], 1)
        self.assertFalse(first["entire_manifest_complete"])
        self.assertEqual(calls, [100, 101])
        second = collect_sources(config(), archive,
                                 lambda item: (source_url(item), 200, snapshot(item)), Memory(), 5,
                                 sleep=lambda _: None)
        self.assertEqual((second["cache_hits"], second["downloaded"]), (1, 1))
        self.assertTrue(second["entire_manifest_complete"])
        third = collect_sources(config(), archive,
                                lambda _: self.fail("Cached run made a request"), Memory(), 5)
        self.assertEqual(third["network_navigations"], 0)
        self.assertEqual(third["cache_hits"], 2)

    def test_request_spacing_and_limit_do_not_claim_entire_manifest(self):
        archive = Archive(self.root, config())
        tick = [0.0]
        starts = []
        def fetch(item):
            starts.append(tick[0])
            tick[0] += .2
            return source_url(item), 200, snapshot(item)
        def sleep(seconds):
            tick[0] += seconds
        first = collect_sources(config(), archive, fetch, Memory(), 1, clock=lambda: tick[0], sleep=sleep)
        self.assertEqual(first["status"], "selected_batch_complete")
        self.assertFalse(first["entire_manifest_complete"])
        # Different archive makes both tasks uncached to exercise interval.
        collect_sources(config(), Archive(self.root / "other", config()), fetch, Memory(), 2,
                        clock=lambda: tick[0], sleep=sleep)
        self.assertGreaterEqual(starts[2] - starts[1], 7)

    def test_memory_guard_stops_before_fetch(self):
        class Exhausted:
            def check(self):
                raise SourceError("memory_budget_exceeded")
        result = collect_sources(config(), Archive(self.root, config()),
                                 lambda _: self.fail("Memory guard ignored"), Exhausted(), 5)
        self.assertEqual(result["network_navigations"], 0)
        self.assertEqual(result["error"], "memory_budget_exceeded")

    def test_corrupt_resume_reports_incomplete_without_network_or_repair(self):
        archive = Archive(self.root, config())
        item = task()
        text, identity = validate_snapshot(item, source_url(item), 200, snapshot(item))
        archive.save(item, text, identity)
        source, _ = archive.paths(item)
        source.write_bytes(b"corrupt")
        result = collect_sources(config(), archive, lambda _: self.fail("Corrupt cache refetched"), Memory(), 5)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["error"], "source_checksum_mismatch")
        self.assertFalse(result["entire_manifest_complete"])
        self.assertEqual(source.read_bytes(), b"corrupt")

    def test_profile_lock_prevents_parallel_login_or_collection(self):
        lock = self.root / "lock"
        with exclusive_file(lock):
            with self.assertRaises(SourceError):
                with exclusive_file(lock):
                    self.fail("Two active collectors")
        self.assertFalse(lock.exists())

    def test_manifest_rejects_duplicates_unsafe_handles_and_invalid_intervals(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(config()), encoding="utf-8")
        self.assertEqual(read_config(path), config())
        for bad in [config() | {"submissions": [task(), task()]},
                    config() | {"request_interval_seconds": 1},
                    config() | {"run_id": "../escape"},
                    config() | {"submissions": [task() | {"expected_handle": "../../escape"}]}]:
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
