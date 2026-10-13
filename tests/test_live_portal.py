import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cf_pilot.benchmark import FEATURE_COLUMNS, behavior_features
from cf_pilot.live_portal import CACHE, UpstreamError, contest_id, get_live_portal, prepare_contest


class FakePolicy:
    def __init__(self, payload, snapshot, x):
        self.snapshot, self.x = snapshot, x


class LivePortalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.payload = Path(self.tmp.name)
        pre = {"feature_version": "behaviour_only_v1", "columns": list(FEATURE_COLUMNS),
               "stats": {name: {"impute": 0, "mean": 0, "scale": 1} for name in FEATURE_COLUMNS}}
        (self.payload / "preprocessor.json").write_text(json.dumps(pre))
        (self.payload / "model.json").write_text('{"selected":"frozen"}')
        self.contest = {"id": 2274, "phase": "FINISHED", "startTimeSeconds": 1800000000, "durationSeconds": 7200}
        CACHE.clear()

    def tearDown(self):
        self.tmp.cleanup()
        CACHE.clear()

    def submission(self, sid, handle="two", problem="A", seconds=60, kind="CONTESTANT"):
        return {"id": sid, "contestId": 2274, "creationTimeSeconds": self.contest["startTimeSeconds"] + seconds,
                "author": {"members": [{"handle": handle}], "participantType": kind},
                "problem": {"index": problem}, "programmingLanguage": "GNU C++17", "verdict": "OK"}

    def test_scope_and_verdict_independence(self):
        items = [self.submission(1), self.submission(2, problem="B", seconds=100),
                 self.submission(3, handle="one"), self.submission(4, handle="practice", kind="PRACTICE"),
                 self.submission(5, handle="late", seconds=7200)]
        with patch("cf_pilot.live_portal.RemotePortal", FakePolicy):
            first = prepare_contest(self.payload, self.contest, items)
            for item in items:
                item["verdict"] = "SKIPPED"
                item["passedTestCount"] = 1000
            second = prepare_contest(self.payload, self.contest, items)
        self.assertEqual(len(first.directory), 2)
        self.assertEqual([row["handle"] for row in first.snapshot["rows"]], ["two"])
        self.assertFalse(first.by_id["2274:one"]["eligible"])
        np.testing.assert_array_equal(first.x, second.x)
        self.assertNotEqual(first.snapshot["snapshot_id"], second.snapshot["snapshot_id"])
        self.assertEqual(first.x[0, 0], behavior_features(items[:2], self.contest)["attempts"])

    def test_reject_unfinished_old_and_duplicate(self):
        with self.assertRaises(ValueError):
            prepare_contest(self.payload, dict(self.contest, phase="CODING"), [])
        with self.assertRaises(ValueError):
            prepare_contest(self.payload, dict(self.contest, startTimeSeconds=1600000000), [])
        item = self.submission(1)
        with self.assertRaises(UpstreamError):
            prepare_contest(self.payload, self.contest, [item, item])
        for value in (None, "abc", "https://evil.test", "0", "-1", "999999999"):
            with self.assertRaises(ValueError):
                contest_id(value)

    def test_complete_pagination_and_warm_cache(self):
        items = [self.submission(1), self.submission(2, problem="B")]
        with patch("cf_pilot.live_portal.PAGE_SIZE", 2), patch("cf_pilot.live_portal.RemotePortal", FakePolicy), \
             patch("cf_pilot.live_portal.cf_request", side_effect=[{"contest": self.contest}, items, []]) as request:
            first = get_live_portal(self.payload, 2274)
            second = get_live_portal(self.payload, 2274)
        self.assertIs(first, second)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[-1].args[1]["from"], 3)

    def test_upstream_failure_never_caches_partial_data(self):
        with patch("cf_pilot.live_portal.cf_request", side_effect=[{"contest": self.contest}, UpstreamError("blocked")]):
            with self.assertRaises(UpstreamError):
                get_live_portal(self.payload, 2274)
        self.assertNotIn(2274, CACHE)

    def test_old_contest_rejected_before_fetching_submission_pages(self):
        old = dict(self.contest, startTimeSeconds=1600000000)
        with patch("cf_pilot.live_portal.cf_request", return_value={"contest": old}) as request:
            with self.assertRaisesRegex(ValueError, "2025 onward"):
                get_live_portal(self.payload, 2274)
        self.assertEqual(request.call_count, 1)
        self.assertNotIn(2274, CACHE)


if __name__ == "__main__":
    unittest.main()
