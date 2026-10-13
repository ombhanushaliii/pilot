from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock, Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np

from cf_pilot.portal import Portal, handler_for
from cf_pilot.benchmark import FEATURE_GROUPS
from cf_pilot.rl_data import binary_metrics


class PortalTests(unittest.TestCase):
    def fixture(self):
        portal = Portal.__new__(Portal)
        portal.contest_id = 2274
        portal.contest = {"name": "Fixture", "startTimeSeconds": 1, "durationSeconds": 7200}
        portal.snapshot_time = 2
        portal.selection = {"selected": "saved", "selected_update": 12000}
        portal.rows = [{"id": "2274:"+name, "handle": name, "attempts": 2, "problem_count": 2, "last_seconds": 100,
                        "status": "unchecked", "actions": [], "inspections": None, "model_index": idx,
                        "submissions": [], "features": {n: 0 for names in FEATURE_GROUPS.values() for n in names}}
                       for idx, name in enumerate(("zebra", "alpha", "middle"))]
        portal.by_id = {r["id"]:r for r in portal.rows}
        portal.data = {"x": np.zeros((3,48), dtype=np.float32), "y": np.array([1,0,0])}
        portal.network = object()
        portal.expected = binary_metrics(portal.data["y"], [1,0,1], [0,2,1])
        portal.lock = Lock()
        portal.reviewed = False
        portal.elapsed = None
        portal.review_runs = 0
        return portal

    def test_policy_receives_features_only_and_repeat_request_is_cached(self):
        portal = self.fixture()
        result = (np.array([1,0,1]), np.array([0,2,1]), np.array([0,3,1]), [[4],[0,1,3],[0,4]])
        with patch('cf_pilot.portal.infer', return_value=result) as inference:
            first = portal.run_review()
            second = portal.run_review()
        self.assertIs(inference.call_args.args[1], portal.data['x'])
        self.assertEqual(len(inference.call_args.args), 2)
        self.assertEqual(inference.call_count, 1)
        self.assertFalse(first['cached'])
        self.assertTrue(second['cached'])
        self.assertEqual(portal.review_runs, 1)
        self.assertEqual(first['flagged'], 2)
        self.assertEqual([r['handle'] for r in portal.participants()['rows']], ['middle','zebra','alpha'])
        self.assertEqual(portal.participants(status='review')['total'], 2)
        self.assertEqual(portal.detail('2274:alpha')['actions'], ['inspect_pacing','inspect_trajectory','low_priority'])

    def test_failed_reproduction_does_not_publish_or_cache_decisions(self):
        portal = self.fixture()
        with patch('cf_pilot.portal.infer', return_value=(np.array([0,0,0]),np.zeros(3),np.zeros(3),[[3]]*3)):
            with self.assertRaises(ValueError):
                portal.run_review()
        self.assertFalse(portal.reviewed)
        self.assertEqual(portal.review_runs, 0)
        self.assertTrue(all(r['status']=='unchecked' for r in portal.rows))

    def test_filters_and_pagination_keep_identifiers_out_of_model_features(self):
        portal = self.fixture()
        self.assertEqual(portal.participants(query='ALP')['rows'][0]['handle'], 'alpha')
        self.assertEqual(portal.participants(page=2,page_size=1)['rows'][0]['handle'], 'middle')
        self.assertIsNone(portal.overview()['flagged'])
        for options in ({'status':'cheater'}, {'page':0}, {'page_size':101}):
            with self.assertRaises(ValueError):
                portal.participants(**options)

    def test_http_serves_only_allowlisted_files_and_rejects_cross_origin_actions(self):
        portal = self.fixture()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'index.html').write_text('<h1>Portal</h1>')
            (root/'secret.txt').write_text('must not be served')
            server = ThreadingHTTPServer(('127.0.0.1',0),handler_for(portal,root))
            thread = Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base = 'http://127.0.0.1:'+str(server.server_port)
            try:
                self.assertIn(b'Portal',urlopen(base).read())
                with self.assertRaises(HTTPError) as err:
                    urlopen(base+'/secret.txt')
                self.assertEqual(err.exception.code,404)
                with self.assertRaises(HTTPError) as err:
                    urlopen(Request(base+'/api/review',method='POST',headers={'Origin':'https://another-site.test'}))
                self.assertEqual(err.exception.code,403)
                with self.assertRaises(HTTPError) as err:
                    urlopen(Request(base+'/api/contest',headers={'Host':'another-site.test'}))
                self.assertEqual(err.exception.code,403)
                with self.assertRaises(HTTPError) as err:
                    urlopen(base+'/api/participants?page=0')
                self.assertEqual(err.exception.code,400)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
