"""Build a self-contained bookmarklet. Reads loaded DOM; makes no requests."""

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

from cf_pilot.sources import PROJECT, SNAPSHOT_JS, read_config


def build_script(config):
    tasks = json.dumps(config["submissions"], separators=(",", ":"))
    return """(() => { try {
      if (location.protocol !== 'https:' || location.host !== 'codeforces.com') throw new Error('Open a Codeforces submission page first.');
      const match = location.pathname.match(/^\\/contest\\/(\\d+)\\/submission\\/(\\d+)$/);
      if (!match) throw new Error('This is not a supported submission URL.');
      const tasks = TASKS;
      const task = tasks.find(x => x.contest_id === Number(match[1]) && x.submission_id === Number(match[2]));
      if (!task) throw new Error('This submission is outside the five-case pilot.');
      const snapshot = (SNAPSHOT)();
      if (snapshot.challenge || /just a moment|verify you are human/i.test(snapshot.title)) throw new Error('The source page has not loaded normally.');
      if (snapshot.source_count !== 1 || !snapshot.source || !snapshot.source.trim()) throw new Error('No unique source code element was found.');
      const url = 'https://codeforces.com' + location.pathname;
      snapshot.rows = snapshot.rows.filter(x => new URL(x.submission_href, url).href === url);
      if (snapshot.rows.length !== 1) throw new Error('Submission table identity is missing or ambiguous.');
      const row = snapshot.rows[0];
      if (row.authors.length !== 1 || row.authors[0].toLowerCase() !== task.expected_handle.toLowerCase()) throw new Error('Submission author does not match the pilot manifest.');
      const problem = new URL(row.problem_href || '', url);
      const p = problem.pathname.match(/^\\/contest\\/(\\d+)\\/problem\\/([A-Za-z0-9]+)$/);
      if (problem.origin !== 'https://codeforces.com' || !p || Number(p[1]) !== task.contest_id || !row.language) throw new Error('Problem or language identity is missing.');
      if (task.expected_problem && task.expected_problem !== p[2]) throw new Error('Problem differs from the pilot manifest.');
      snapshot.title = ''; // Do not retain unrelated page-title content.
      const value = {version: 1, method: 'user_click_loaded_dom', source_url: url, exported_at_utc: new Date().toISOString(), snapshot};
      const blob = new Blob([JSON.stringify(value, null, 2)], {type: 'application/json;charset=utf-8'});
      const local = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = local;
      link.download = 'cf-source-' + task.contest_id + '-' + task.submission_id + '.json';
      document.body.appendChild(link);
      link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(local), 2000);
    } catch (error) { alert('Source export stopped: ' + error.message); } })();""".replace("TASKS", tasks).replace("SNAPSHOT", SNAPSHOT_JS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(PROJECT / "config" / "source-browser-export.json")
    script = build_script(config)
    bookmark = "javascript:" + quote(script, safe="")
    links = "".join(f'<li><a href="https://codeforces.com/contest/{x["contest_id"]}/submission/{x["submission_id"]}">{html.escape(x["expected_handle"])}: {x["submission_id"]}</a></li>' for x in config["submissions"])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Codeforces pilot source export</title>
    <style>body{{font:17px/1.6 system-ui;max-width:760px;margin:40px auto;padding:0 24px}}a{{color:#095aa8}}.button{{display:inline-block;padding:12px 20px;background:#eef5ff;border:1px solid #accbea;border-radius:8px}}code{{background:#eee;padding:2px 5px}}</style>
    <h1>Export a source already open in Helium</h1>
    <p>Drag the link below to your bookmarks bar. Open a pilot submission normally, wait until you can see its code, then click that bookmark.</p>
    <p><a class="button" href="{html.escape(bookmark, quote=True)}">Export CF pilot source v2</a></p>
    <p>This reads the loaded code and submission table and downloads a JSON file. It makes no page requests, reads no cookies and launches no browser. The importer checks the saved identity. When syntax highlighting splits code into line elements, the exporter reconstructs line breaks and records that method; original uploaded whitespace is not guaranteed. Labels stay unknown.</p>
    <p>Capture this first: Thinkin, submission 368575830.</p><ul>{links}</ul>
    <p>Downloads are named <code>cf-source-CONTEST-SUBMISSION.json</code>. Keep them local. You can inspect the bookmark's complete code in the accompanying <code>source-export-bookmarklet.js</code> file.</p>
    <p>Clicking the button on this setup page will stop with an instruction to open Codeforces. Use the saved bookmark on the submission page. If your browser blocks bookmarklets, use a manually saved HTML page instead; do not change its security settings.</p>
    <p>This is a five-case extraction pilot, not an automated crawler or a test of model accuracy. Live extraction in Helium is awaiting the first export.</p></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    args.output.with_name("source-export-bookmarklet.js").write_text(script, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
