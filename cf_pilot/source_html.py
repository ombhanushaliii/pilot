"""Extract a user-saved submission page offline. HTML scripts are never run."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from .sources import (Archive, PROJECT, SourceError, exclusive_file, read_config,
                      source_url, utc_now, validate_loaded_snapshot)


class SavedPageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = []
        self.in_title = False
        self.source_depth = 0
        self.sources = []
        self.tables = []
        self.stack = []
        self.challenge = False
        self.rendered_lines = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") == "challenge-form" or "cf-turnstile" in attrs.get("class", "").split():
            self.challenge = True
        if tag == "title":
            self.in_title = True
        if tag == "pre" and attrs.get("id") == "program-source-text":
            self.sources.append([])
            self.source_depth = 1
        elif self.source_depth and tag not in {"br", "img", "input", "meta", "link", "hr", "wbr"}:
            self.source_depth += 1
        if self.source_depth and tag == "ol" and "linenums" in attrs.get("class", "").split():
            self.rendered_lines = True
        if tag == "table":
            self.stack.append({"rows": [], "row": None, "cell": None})
        elif self.stack:
            table = self.stack[-1]
            if tag == "tr":
                table["row"] = []
                table["rows"].append(table["row"])
            elif tag in ("th", "td") and table["row"] is not None:
                table["cell"] = {"tag": tag, "text": [], "hrefs": []}
                table["row"].append(table["cell"])
            elif tag == "a" and table["cell"] is not None and "href" in attrs:
                table["cell"]["hrefs"].append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if self.source_depth and tag not in {"br", "img", "input", "meta", "link", "hr", "wbr"}:
            self.source_depth -= 1
        if tag == "table" and self.stack:
            self.tables.append(self.stack.pop()["rows"])
        elif self.stack and tag in ("td", "th"):
            self.stack[-1]["cell"] = None
        elif self.stack and tag == "tr":
            self.stack[-1]["row"] = None

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.source_depth:
            self.sources[-1].append(data)
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["text"].append(data)

    def snapshot(self):
        rows, notices = [], {}
        for table in self.tables:
            headers = next((row for row in table if any(cell["tag"] == "th" for cell in row)), [])
            names = ["".join(cell["text"]).strip().casefold() for cell in headers]
            def column(choices):
                return next((i for i, name in enumerate(names) if name in choices), -1)
            who, lang, identifier = column({"who", "author"}), column({"lang", "language"}), column({"#", "id"})
            if min(who, lang, identifier) < 0:
                continue
            for row in table:
                if len(row) <= max(who, lang, identifier) or any(cell["tag"] != "td" for cell in row):
                    continue
                sid = "".join(row[identifier]["text"]).strip()
                if not re.fullmatch(r"[1-9][0-9]*", sid):
                    continue
                hrefs = [href for cell in row for href in cell["hrefs"]]
                problems = [href for href in hrefs if re.fullmatch(r"/contest/\d+/problem/[A-Za-z0-9]+", urlsplit(href).path)
                            and urlsplit(href).netloc in ("", "codeforces.com")]
                if len(problems) != 1:
                    continue
                problem = problems[0]
                contest = urlsplit(problem).path.split("/")[2]
                authors = [unquote(urlsplit(href).path[len("/profile/"):]) for href in row[who]["hrefs"]
                           if urlsplit(href).path.startswith("/profile/") and urlsplit(href).netloc in ("", "codeforces.com")]
                url = f"/contest/{contest}/submission/{sid}"
                rows.append({"submission_href": url, "authors": authors,
                             "language": "".join(row[lang]["text"]).strip(), "problem_href": problem})
                notice = "".join(row[who]["text"])
                if "out of competition" in notice.casefold():
                    notices[url] = "Out of competition"
        return {"title": "".join(self.title), "challenge": self.challenge, "source_count": len(self.sources),
                "source": "".join(self.sources[0]) if len(self.sources) == 1 else None, "rows": rows}, notices


def load_html(path, task):
    with Path(path).open("rb") as stream:
        body = stream.read(10 * 1024 * 1024 + 1)
    if len(body) > 10 * 1024 * 1024:
        raise SourceError("saved_html_too_large")
    try:
        # Normalize HTML newlines as browsers do before producing DOM text.
        text = body.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError:
        raise SourceError("saved_html_not_utf8") from None
    parser = SavedPageParser()
    parser.feed(text)
    parser.close()
    if parser.rendered_lines:
        raise SourceError("saved_html_contains_rendered_lines_use_browser_export")
    snapshot, notices = parser.snapshot()
    source, identity = validate_loaded_snapshot(task, source_url(task), snapshot)
    title_id = re.search(r"Submission\s+#(\d+)", snapshot["title"])
    if not title_id or int(title_id[1]) != task["submission_id"]:
        raise SourceError("saved_html_title_identity_mismatch")
    identity.update({"acquisition_method": "user_saved_html", "imported_at_utc": utc_now(),
                     "saved_html_sha256": hashlib.sha256(body).hexdigest(), "http_status_observed": False,
                     "extraction": "saved HTML parsed offline; entities decoded; browser-style LF newlines",
                     "capture_time_known": False,
                     "page_participation_notice": notices.get(urlsplit(source_url(task)).path)})
    return source, identity


def import_html(path, task, archive):
    source, identity = load_html(path, task)
    with exclusive_file(archive.root / ".import-lock"):
        if archive.cached(task):
            file, metadata = archive.paths(task)
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if file.read_bytes() != source.encode("utf-8") or any(saved.get(key) != identity[key] for key in ("observed_handle", "problem_index", "language")):
                raise SourceError("saved_html_conflicts_with_existing_source")
            status = "already_saved"
        else:
            archive.save(task, source, identity)
            status = "saved"
        archive.event({"submission_id": task["submission_id"], "status": status, "method": "user_saved_html"})
    return {"submission_id": task["submission_id"], "status": status, "label": "unknown", "network_requests": 0,
            "observed_handle": identity["observed_handle"], "problem_index": identity["problem_index"],
            "language": identity["language"], "page_participation_notice": identity["page_participation_notice"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "config" / "source-browser-export.json")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.manifest)
        tasks = [task for task in config["submissions"] if task["submission_id"] == args.submission_id]
        if len(tasks) != 1:
            raise SourceError("submission_not_in_manifest")
        load_html(args.html, tasks[0])
        archive = Archive(PROJECT / "data" / "sources" / config["run_id"], config)
        print(json.dumps(import_html(args.html, tasks[0], archive), indent=2))
        return 0
    except (SourceError, ValueError, OSError) as error:
        print(f"Saved HTML import stopped: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
