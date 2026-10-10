import gzip
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from cf_pilot.api import Client, CollectionError
from cf_pilot.cohort import Candidate, parse_organization_page, select_sample, validate_organization_url
from cf_pilot.collect import DEFAULTS, choose_contests, collect_pages
from cf_pilot.collect import collect
from cf_pilot.audit import audit_run, write_reports
from cf_pilot.sessions import eligibility, prior_sessions, build_sessions
from cf_pilot.store import Store, canonical


def submission(identifier, created=110, handle="alice", kind="CONTESTANT", contest_id=1):
    return {"id": identifier, "contestId": contest_id, "creationTimeSeconds": created,
            "relativeTimeSeconds": created - 100, "problem": {"contestId": contest_id, "index": "A"},
            "author": {"participantType": kind, "members": [{"handle": handle}]},
            "programmingLanguage": "GNU C++17", "verdict": "WRONG_ANSWER"}


def ok(result):
    return json.dumps({"status": "OK", "result": result}).encode()


class Clock:
    def __init__(self):
        self.now = 100.0
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "work" / "test_tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.store = Store(Path(self.temp.name))
        self.store.start_run("test", DEFAULTS)
        self.clock = Clock()
    def tearDown(self):
        self.store.close()
        if not Path(self.temp.name).resolve().is_relative_to(self.temp_root.resolve()):
            raise AssertionError("Temporary cleanup target escaped test workspace")
        self.temp.cleanup()
    def client(self, transport, **kwargs):
        return Client(self.store, "test", transport=transport, clock=self.clock.time,
                      sleep=self.clock.sleep, **kwargs)

    def test_organisation_excludes_sidebar_and_handles_unclosed_rows(self):
        html = '''<a href="/profile/tourist">tourist</a>
        <div class="datatable ratingsDatatable"><table>
        <tr><th>Who</th><th>=</th></tr>
        <tr><td><a href="/profile/alice">alice</a></td><td>1400</td>
        <tr><td><a href="/profile/bob">bob</a></td><td>800</td></tr>
        </table></div><a href="/profile/admin">admin</a>'''
        candidates, pages = parse_organization_page(html, "https://codeforces.com/ratings/organization/5276")
        self.assertEqual(candidates, [Candidate("alice", 1400), Candidate("bob", 800)])
        self.assertEqual(pages, [])

    def test_missing_table_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_organization_page('<a href="/profile/alice">alice</a>', "https://codeforces.com/ratings/organization/5276")

    def test_reproducible_sampling(self):
        items = [Candidate(f"p{x}", x) for x in range(20)]
        self.assertEqual(select_sample(items, 8, 736), select_sample(list(reversed(items)), 8, 736))
        self.assertEqual(len(select_sample(items, 8, 736)), 8)

    def test_url_restriction(self):
        for url in ["http://codeforces.com/ratings/organization/1", "https://evil.test/ratings/organization/1",
                    "https://codeforces.com@evil.test/ratings/organization/1", "https://codeforces.com/ratings/organization/1?x=1"]:
            with self.assertRaises(ValueError):
                validate_organization_url(url)

    def test_cache_prevents_rerun_network_calls(self):
        calls = []
        client = self.client(lambda url, timeout: calls.append(url) or ok([1, 2]))
        first, key = client.call("user.rating", handle="alice")
        second, _ = client.call("user.rating", handle="alice")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.cache_hits, 1)
        self.assertEqual(self.store.cached("test", key), ok([1, 2]))

    def test_global_spacing_across_clients(self):
        times = []
        transport = lambda url, timeout: times.append(self.clock.time()) or ok([])
        self.client(transport).call("user.rating", handle="alice")
        self.client(transport).call("user.rating", handle="bob")
        self.assertGreaterEqual(times[1] - times[0], 2.1 - 1e-8)

    def test_api_rate_failure_retried_and_never_cached(self):
        responses = [json.dumps({"status": "FAILED", "comment": "Call limit exceeded"}).encode(), ok([])]
        client = self.client(lambda url, timeout: responses.pop(0))
        client.call("user.rating", handle="alice")
        self.assertEqual(client.network_requests, 2)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM responses").fetchone()[0], 1)

    def test_permanent_error_is_not_an_empty_success(self):
        calls = []
        client = self.client(lambda url, timeout: calls.append(url) or json.dumps(
            {"status": "FAILED", "comment": "handle: User with handle alice not found"}).encode())
        with self.assertRaises(CollectionError):
            client.call("user.rating", handle="alice")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM responses").fetchone()[0], 0)

    def test_http_403_is_not_bypassed(self):
        def denied(url, timeout):
            raise HTTPError(url, 403, "Forbidden", None, None)
        client = self.client(denied)
        with self.assertRaises(CollectionError):
            client.call("user.rating", handle="alice")
        self.assertEqual(client.network_requests, 1)

    def test_corrupted_archive_stops_without_refetch(self):
        client = self.client(lambda url, timeout: ok([]))
        _, key = client.call("user.rating", handle="alice")
        path = self.store.db.execute("SELECT raw_path FROM responses WHERE request_key=?", (key,)).fetchone()[0]
        (self.store.root / path).write_bytes(gzip.compress(b"changed"))
        with self.assertRaises(ValueError):
            client.call("user.rating", handle="alice")
        self.assertEqual(client.network_requests, 1)

    def test_overlap_deduplication_and_completed_job_rerun(self):
        pages = {1: [9, 8, 7], 3: [7, 6, 5], 5: [5, 4]}
        calls = []
        def transport(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            calls.append(index)
            return ok([submission(x) for x in pages[index]])
        client = self.client(transport)
        self.assertTrue(collect_pages(client, "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 5))
        self.assertEqual(calls, [1, 3, 5])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM submissions").fetchone()[0], 6)
        collect_pages(client, "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 5)
        self.assertEqual(calls, [1, 3, 5])
        self.store.ingest_submissions("test", "same", [submission(9)])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM submissions").fetchone()[0], 6)

    def test_interrupted_job_resumes_from_checkpoint(self):
        pages = {1: [9, 8, 7], 3: [7, 6]}
        def interrupted(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            if index == 3:
                raise RuntimeError("simulated crash")
            return ok([submission(x) for x in pages[index]])
        with self.assertRaises(CollectionError):
            collect_pages(self.client(interrupted), "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 5)
        self.assertEqual(self.store.job("test", "session:alice:1")["next_from"], 3)
        calls = []
        def resumed(url, timeout):
            index = int(parse_qs(urlsplit(url).query)["from"][0])
            calls.append(index)
            return ok([submission(x) for x in pages[index]])
        self.assertTrue(collect_pages(self.client(resumed), "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 5))
        self.assertEqual(calls, [3])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM submissions").fetchone()[0], 4)

    def test_page_cap_is_incomplete(self):
        client = self.client(lambda url, timeout: ok([submission(x) for x in [9, 8, 7]]))
        self.assertFalse(collect_pages(client, "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 1))
        self.assertEqual(self.store.job("test", "session:alice:1")["status"], "capped")

    def test_wrong_author_rejected(self):
        client = self.client(lambda url, timeout: ok([submission(1, handle="bob")]))
        with self.assertRaises(CollectionError):
            collect_pages(client, "session:alice:1", "contest.status", {"contestId": 1, "handle": "alice"}, 3, 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM submissions").fetchone()[0], 0)

    def test_changed_snapshot_recorded_without_overwriting(self):
        self.store.ingest_submissions("test", "a", [submission(1)])
        self.store.ingest_submissions("test", "b", [submission(1) | {"verdict": "SKIPPED"}])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM submission_conflicts").fetchone()[0], 1)
        original = json.loads(self.store.db.execute("SELECT payload_json FROM submissions").fetchone()[0])
        self.assertEqual(original["verdict"], "WRONG_ANSWER")

    def test_run_config_is_frozen(self):
        with self.assertRaises(ValueError):
            self.store.start_run("test", DEFAULTS | {"seed": 99})

    def test_participation_and_time_eligibility(self):
        contest = {"id": 1, "startTimeSeconds": 100, "durationSeconds": 100}
        self.assertTrue(eligibility(submission(1), contest)[0])
        for kind in ["PRACTICE", "VIRTUAL", "OUT_OF_COMPETITION", "MANAGER"]:
            self.assertFalse(eligibility(submission(1, kind=kind), contest)[0])
        self.assertFalse(eligibility(submission(1, created=200), contest)[0])
        self.assertFalse(eligibility(submission(1) | {"author": {"ghost": True}}, contest)[0])

    def test_future_data_cannot_change_prior_context(self):
        earlier = {"id": 1, "startTimeSeconds": 100, "durationSeconds": 100}
        target = {"id": 2, "startTimeSeconds": 300, "durationSeconds": 100}
        future = {"id": 3, "startTimeSeconds": 500, "durationSeconds": 100}
        records = [submission(1, created=110, contest_id=1)]
        before = prior_sessions(records, {1: earlier, 2: target}, target, 180)
        after = prior_sessions(records + [submission(2, created=510, contest_id=3)],
                               {1: earlier, 2: target, 3: future}, target, 180)
        self.assertEqual(before, [1])
        self.assertEqual(before, after)

    def test_only_finished_contests_selected(self):
        contests = [{"id": x, "name": "Round (Div. 2)", "type": "CF", "phase": phase,
                     "startTimeSeconds": 100 + x, "durationSeconds": 100}
                    for x, phase in [(1, "FINISHED"), (2, "FINISHED"), (3, "CODING")]]
        config = DEFAULTS | {"contest_count": 2, "min_contest_age_days": 0}
        selected = choose_contests(contests, [{"contestId": 1}, {"contestId": 2}, {"contestId": 3}], config, 10000)
        self.assertEqual([item["id"] for item in selected], [1, 2])

    def test_rebuild_preserves_labels_and_never_infers_them(self):
        contest = {"id": 1, "startTimeSeconds": 100, "durationSeconds": 100}
        self.store.db.execute("INSERT INTO contests VALUES(?,?,?,1)", ("test", 1, canonical(contest)))
        self.store.db.execute("INSERT INTO cohort VALUES(?,?,?,?,?)", ("test", "alice", 900, 1, "source"))
        self.store.db.commit()
        self.store.ingest_submissions("test", "a", [submission(1)])
        build_sessions(self.store, "test", 180)
        self.assertEqual(self.store.db.execute("SELECT label FROM labels").fetchone()[0], "unknown")
        self.store.db.execute("UPDATE labels SET label='substantiated_misconduct'")
        self.store.db.commit()
        build_sessions(self.store, "test", 180)
        self.assertEqual(self.store.db.execute("SELECT label FROM labels").fetchone()[0], "substantiated_misconduct")

    def test_end_to_end_collection_and_offline_rerun(self):
        config = DEFAULTS | {"sample_size": 2, "contest_count": 1, "contest_ids": [1]}
        self.store.start_run("integrated", config)
        now = int(self.store.db.execute("SELECT created_at FROM runs WHERE run_id='integrated'").fetchone()[0])
        target = {"id": 1, "name": "Round (Div. 2)", "phase": "FINISHED", "type": "CF",
                  "startTimeSeconds": now - 30 * 86400, "durationSeconds": 7200}
        prior = target | {"id": 2, "startTimeSeconds": now - 60 * 86400}
        records = {handle: [submission(3 + offset, created=target["startTimeSeconds"] + 30, handle=handle, contest_id=1),
                            submission(2 + offset, created=prior["startTimeSeconds"] + 30, handle=handle, contest_id=2)]
                   for handle, offset in [("alice", 0), ("bob", 100)]}
        html = '<div class="ratingsDatatable"><table>' + ''.join(
            f'<tr><td><a href="/profile/{handle}">{handle}</a></td><td>1000</td></tr>'
            for handle in records) + '</table></div>'
        def transport(url, timeout):
            parsed = urlsplit(url)
            q = parse_qs(parsed.query)
            method = parsed.path.split('/')[-1]
            if parsed.path.startswith('/ratings/'):
                return html.encode()
            if method == 'contest.list':
                return ok([target, prior])
            if method == 'user.rating':
                return ok([{"contestId": 1, "handle": q['handle'][0]}])
            if method == 'contest.standings':
                self.assertEqual(set(q), {'contestId'})
                return ok({"contest": target, "rows": [{"party": item[0]['author']} for item in records.values()]})
            if method == 'contest.status':
                return ok([records[q['handle'][0]][0]])
            if method == 'user.status':
                return ok(records[q['handle'][0]])
            raise AssertionError(url)
        client = Client(self.store, 'integrated', transport=transport, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertTrue(collect(client, config, progress=lambda text: None))
        build_sessions(self.store, 'integrated', 180)
        self.store.db.execute("UPDATE runs SET status='complete' WHERE run_id='integrated'")
        self.store.db.commit()
        report = audit_run(self.store, 'integrated')
        self.assertTrue(report['collection_checks_passed'], report['issues'])
        self.assertEqual(report['counts']['eligible_sessions'], 2)
        self.assertEqual(report['counts']['eligible_sessions_with_prior_history'], 2)
        self.assertFalse(report['training_ready'])
        before = report['counts'].copy()
        calls = client.network_requests
        self.assertTrue(collect(client, config, progress=lambda text: None))
        build_sessions(self.store, 'integrated', 180)
        self.assertEqual(client.network_requests, calls)
        self.assertEqual(audit_run(self.store, 'integrated')['counts'], before)
        output = Path(self.temp.name) / 'reports'
        write_reports(self.store, 'integrated', output, report)
        worksheet = output / 'labels_for_review.csv'
        worksheet.write_text('manual review notes', encoding='utf-8')
        write_reports(self.store, 'integrated', output, report)
        self.assertEqual(worksheet.read_text(encoding='utf-8'), 'manual review notes')


if __name__ == "__main__":
    unittest.main()
