import json
from pathlib import Path
import tempfile
import unittest

from cf_pilot.source_html import import_html, load_html
from cf_pilot.sources import Archive, PROJECT, SourceError
from test_sources import config, task


def page():
    # Minimal regression layout from the saved page, with fictional code.
    return '''<!doctype html><title>Submission #100 - Codeforces</title>
    <table><tr><th>#</th><th>Author</th><th>Problem</th><th>Lang</th></tr>
    <tr><td>100</td><td>Out of competition:<br><a href="/profile/Example">Example</a></td>
    <td><a href="/contest/2211/problem/E">2211E</a></td><td>C++20 (GCC 13-64)</td></tr></table>
    <pre id="program-source-text">  // λ\nif (x &lt; 2 &amp;&amp; y &gt; 0) {\n\treturn 0;\n}\n\n</pre>
    <script>throw new Error('This script must never run');</script>'''


class SavedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "work")
        self.root = Path(self.temp.name)
        self.file = self.root / "saved.html"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, text):
        self.file.write_bytes(text.encode("utf-8"))

    def test_saved_plain_id_html_entities_newlines_and_unknown_label(self):
        self.write(page().replace("\n", "\r\n"))
        source, identity = load_html(self.file, task())
        self.assertEqual(source, "  // λ\nif (x < 2 && y > 0) {\n\treturn 0;\n}\n\n")
        self.assertEqual(identity["problem_index"], "E")
        self.assertEqual(identity["page_participation_notice"], "Out of competition")
        self.assertFalse(identity["capture_time_known"])
        archive = Archive(self.root / "archive", config())
        self.assertEqual(import_html(self.file, task(), archive)["status"], "saved")
        self.assertEqual(import_html(self.file, task(), archive)["status"], "already_saved")
        self.assertEqual(json.loads(archive.paths(task())[1].read_text(encoding="utf-8"))["label"], "unknown")

    def test_incorrect_id_title_author_problem_and_duplicate_table_fail(self):
        for bad in [page().replace("<td>100</td>", "<td>101</td>"),
                    page().replace("Submission #100", "Submission #101"),
                    page().replace("/profile/Example", "/profile/Wrong"),
                    page().replace("/contest/2211/problem/E", "/contest/2212/problem/E"),
                    page() + page()]:
            self.write(bad)
            with self.subTest(bad=bad[:50]), self.assertRaises(SourceError):
                load_html(self.file, task())

    def test_challenge_and_rendered_line_html_fail_visibly(self):
        for bad in [page() + '<form id="challenge-form"></form>',
                    page().replace("  // λ", '<ol class="linenums"><li>line</li></ol>')]:
            self.write(bad)
            with self.assertRaises(SourceError):
                load_html(self.file, task())


if __name__ == "__main__":
    unittest.main()
