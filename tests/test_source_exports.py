import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from cf_pilot.source_exports import import_export, load_export
from cf_pilot.sources import Archive, PROJECT, SourceError, source_url, utc_now
from tools.build_source_exporter import build_script
from test_sources import config, snapshot, task


def export():
    return {"version": 1, "method": "user_click_loaded_dom", "source_url": source_url(task()),
            "exported_at_utc": utc_now(), "snapshot": snapshot(task())}


class SourceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "work")
        self.root = Path(self.temp.name)
        self.path = self.root / "export.json"
        self.settings = config() | {"run_id": "source-export-test-001"}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, value):
        self.path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_offline_import_exact_source_identity_provenance_and_duplicate(self):
        value = export()
        self.write(value)
        archive = Archive(self.root / "archive", self.settings)
        first = import_export(self.path, self.settings, archive)
        self.assertEqual(first, {"submission_id": 100, "status": "saved", "label": "unknown", "network_requests": 0})
        source, meta = archive.paths(task())
        self.assertEqual(source.read_bytes(), value["snapshot"]["source"].encode("utf-8"))
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        self.assertFalse(metadata["http_status_observed"])
        self.assertEqual(metadata["exported_at_utc"], value["exported_at_utc"])
        self.assertEqual(metadata["acquisition_method"], "user_click_loaded_dom")
        self.assertEqual(import_export(self.path, self.settings, archive)["status"], "already_saved")
        value["snapshot"]["source"] += "changed"
        self.write(value)
        with self.assertRaisesRegex(SourceError, "conflicts"):
            import_export(self.path, self.settings, archive)
        self.assertTrue(archive.cached(task()))

    def test_schema_timestamp_and_unexpected_credentials_rejected(self):
        for replacement in [{"cookies": "must not be stored"}, {"version": True},
                            {"exported_at_utc": "2099-01-01T00:00:00Z"},
                            {"exported_at_utc": "2026-10-11T12:00:00"}, {"snapshot": None},
                            {"snapshot": snapshot(task()) | {"cookies": "do not retain"}}]:
            with self.subTest(replacement=replacement):
                self.write(export() | replacement)
                with self.assertRaises(SourceError):
                    load_export(self.path, self.settings)

    def test_wrong_owner_page_or_problem_never_imported(self):
        for field, replacement in [("authors", ["Wrong"]), ("authors", ["Example", "Other"]),
                                   ("problem_href", "/contest/2212/problem/F")]:
            bad = export()
            bad["snapshot"]["rows"][0][field] = replacement
            self.write(bad)
            with self.subTest(field=field), self.assertRaises(SourceError):
                load_export(self.path, self.settings)
        for url in ["https://codeforces.com/enter", "https://elsewhere.example/contest/2211/submission/100", source_url(task(102))]:
            self.write(export() | {"source_url": url})
            with self.subTest(url=url), self.assertRaisesRegex(SourceError, "not_in_manifest"):
                load_export(self.path, self.settings)

    def test_challenge_missing_source_and_corrupt_cached_export_fail(self):
        for replacement in [{"challenge": True}, {"source": "  "}, {"source_count": 2}, {"rows": []}]:
            self.write(export() | {"snapshot": snapshot(task()) | replacement})
            with self.subTest(replacement=replacement), self.assertRaises(SourceError):
                load_export(self.path, self.settings)
        self.write(export())
        archive = Archive(self.root / "archive", self.settings)
        import_export(self.path, self.settings, archive)
        archive.paths(task())[0].write_bytes(b"damaged")
        with self.assertRaisesRegex(SourceError, "checksum"):
            import_export(self.path, self.settings, archive)

    @unittest.skipUnless(shutil.which("node"), "Node unavailable for exporter fixture")
    def test_exporter_script_download_and_failure_paths(self):
        script = self.root / "bookmarklet.js"
        script.write_text(build_script(self.settings), encoding="utf-8")
        result = subprocess.run([shutil.which("node"), str(PROJECT / "tests" / "source_export_fixture.cjs"), str(script)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
