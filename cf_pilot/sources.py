"""Small, sequential Playwright source acquisition. Never assigns misconduct labels.

Run with python -m cf_pilot.sources. Auth state stays in ignored work/.
The public API database and review worksheets are not modified.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

from .store import canonical

PROJECT = Path(__file__).resolve().parent.parent
PROFILE = PROJECT / "work" / "source-browser-profile"
SELECTOR = "#program-source-text"

# Read only the source element and submission table, never cookies, headers,
# account-page HTML, HAR files, or screenshots containing login information.
SNAPSHOT_JS = r"""() => {
  const nodes = [...document.querySelectorAll('#program-source-text')];
  let source = null;
  let sourceTextMode = 'dom_text_content';
  if (nodes.length === 1) {
    const lines = [...nodes[0].querySelectorAll('ol.linenums > li')];
    if (lines.length) {
      // Pretty-printing puts each line in a separate LI. Its textContent
      // concatenates lines, so reconstruct them and record that distinction.
      source = lines.map(x => x.textContent === '\u00a0' ? '' : x.textContent).join('\n');
      sourceTextMode = 'rendered_lines_reconstructed';
    } else {
      source = nodes[0].textContent;
    }
  }
  const rows = [];
  for (const table of document.querySelectorAll('table')) {
    const headers = [...table.querySelectorAll('tr th')].map(x => x.textContent.trim().toLowerCase());
    const who = headers.findIndex(x => x === 'who' || x === 'author');
    const lang = headers.findIndex(x => x === 'lang' || x === 'language');
    const idColumn = headers.findIndex(x => x === '#' || x === 'id');
    if (who < 0 || lang < 0) continue;
    for (const tr of table.querySelectorAll('tr')) {
      const cells = [...tr.children].filter(x => x.tagName === 'TD');
      if (cells.length <= Math.max(who, lang)) continue;
      const submission = [...tr.querySelectorAll('a[href]')].find(x => /\/submission\/\d+(?:$|[?#])/.test(x.getAttribute('href')));
      const problem = [...tr.querySelectorAll('a[href]')].find(x => /\/contest\/\d+\/problem\//.test(x.getAttribute('href')));
      let submissionHref = submission ? submission.getAttribute('href') : null;
      // Submission detail pages render the ID as plain text in the # column.
      // Derive its URL only from that explicit ID and the observed problem URL.
      if (!submissionHref && idColumn >= 0 && cells[idColumn] && problem) {
        const idText = cells[idColumn].textContent.trim();
        const problemURL = new URL(problem.getAttribute('href'), 'https://codeforces.com/');
        const match = problemURL.pathname.match(/^\/contest\/(\d+)\/problem\/[A-Za-z0-9]+$/);
        if (/^[1-9][0-9]*$/.test(idText) && problemURL.origin === 'https://codeforces.com' && match) {
          submissionHref = '/contest/' + match[1] + '/submission/' + idText;
        }
      }
      if (!submissionHref) continue;
      const authors = [...cells[who].querySelectorAll('a[href]')]
        .filter(x => x.getAttribute('href').startsWith('/profile/'))
        .map(x => decodeURIComponent(x.getAttribute('href').slice('/profile/'.length).split(/[?#]/)[0]));
      rows.push({submission_href: submissionHref, authors,
                 language: cells[lang].textContent.trim(), problem_href: problem ? problem.getAttribute('href') : null});
    }
  }
  return {title: document.title, challenge: !!document.querySelector('#challenge-form, .cf-turnstile'),
          source_count: nodes.length, source, source_text_mode: sourceTextMode, rows};
}"""


class SourceError(RuntimeError):
    """Fixed reason codes keep credentials and page contents out of error logs."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_url(task):
    return f"https://codeforces.com/contest/{task['contest_id']}/submission/{task['submission_id']}"


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("version") != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", config.get("run_id", "")):
        raise ValueError("Invalid source run version or ID")
    tasks = config.get("submissions", [])
    if not tasks:
        raise ValueError("No submissions in source manifest")
    seen = set()
    for task in tasks:
        for key in ("contest_id", "submission_id"):
            if type(task.get(key)) is not int or task[key] <= 0:
                raise ValueError("Submission and contest IDs must be positive integers")
        if task["submission_id"] in seen:
            raise ValueError("Duplicate submission ID")
        seen.add(task["submission_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", task.get("expected_handle", "")):
            raise ValueError("A specific expected handle is required")
        if not task.get("purpose"):
            raise ValueError("Acquisition purpose required")
    for key, default, minimum in (("request_interval_seconds", 7, 5),
                                  ("timeout_seconds", 30, 5),
                                  ("min_free_mb", 1024, 0), ("max_process_rss_mb", 1000, 0)):
        config.setdefault(key, default)
        if type(config[key]) not in (int, float) or not minimum <= config[key] <= 3600:
            raise ValueError(f"Invalid {key}")
    return config


def validate_snapshot(task, final_url, status, snapshot):
    if status in (401, 403, 429) or snapshot.get("challenge"):
        raise SourceError("access_challenge_or_denial")
    if status != 200:
        raise SourceError("unexpected_http_status")
    return validate_loaded_snapshot(task, final_url, snapshot)


def validate_loaded_snapshot(task, final_url, snapshot):
    """Validate a loaded DOM; does not claim its HTTP status was observed."""
    parsed = urlsplit(final_url)
    if snapshot.get("challenge"):
        raise SourceError("access_challenge_or_denial")
    if parsed.scheme != "https" or parsed.netloc != "codeforces.com" or parsed.path != urlsplit(source_url(task)).path:
        raise SourceError("unexpected_redirect_or_login_required")
    if snapshot.get("source_count") != 1 or not isinstance(snapshot.get("source"), str) or not snapshot["source"].strip():
        raise SourceError("source_missing_or_ambiguous")
    title = snapshot.get("title", "").casefold()
    if "just a moment" in title or "verify you are human" in title:
        raise SourceError("access_challenge_or_denial")
    rows = []
    for row in snapshot.get("rows", []):
        href = urlsplit(row.get("submission_href", ""))
        if href.netloc and href.netloc != "codeforces.com":
            continue
        if href.path == parsed.path:
            rows.append(row)
    if len(rows) != 1:
        raise SourceError("submission_identity_missing_or_ambiguous")
    row = rows[0]
    if [x.casefold() for x in row.get("authors", [])] != [task["expected_handle"].casefold()]:
        raise SourceError("submission_author_mismatch")
    problem = urlsplit(row.get("problem_href") or "")
    if problem.netloc and problem.netloc != "codeforces.com":
        raise SourceError("problem_identity_mismatch")
    match = re.fullmatch(r"/contest/(\d+)/problem/([A-Za-z0-9]+)", problem.path)
    if not match or int(match[1]) != task["contest_id"] or not row.get("language"):
        raise SourceError("problem_or_language_missing")
    if task.get("expected_problem") and task["expected_problem"] != match[2]:
        raise SourceError("problem_identity_mismatch")
    mode = snapshot.get("source_text_mode", "dom_text_content")
    if mode not in ("dom_text_content", "rendered_lines_reconstructed"):
        raise SourceError("unsupported_source_text_mode")
    identity = {"observed_handle": row["authors"][0], "problem_index": match[2],
                "language": row["language"], "source_text_mode": mode}
    if mode == "rendered_lines_reconstructed":
        identity["extraction"] = "rendered line text joined with LF; uploaded whitespace not guaranteed"
    return snapshot["source"], identity


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_file(path):
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SourceError("collector_or_login_already_running") from None
    try:
        with stream:
            stream.write("Active source collector. Remove only after checking it has stopped.\n")
        yield
    finally:
        path.unlink(missing_ok=True)


class Archive:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        path = self.root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != config:
                raise ValueError("Frozen source manifest differs; use a new run ID")
        else:
            write_json(path, config)

    def paths(self, task):
        stem = f"{task['contest_id']}-{task['submission_id']}"
        return self.root / (stem + ".source.txt"), self.root / (stem + ".json")

    def cached(self, task):
        source, metadata = self.paths(task)
        if not source.exists() and not metadata.exists():
            return False
        if not metadata.exists():
            # An interrupted write cannot be silently trusted or overwritten.
            raise SourceError("orphan_source_file")
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise SourceError("cache_metadata_unreadable") from None
        if not isinstance(value, dict):
            raise SourceError("cache_metadata_unreadable")
        if value.get("task") != task or not source.exists():
            raise SourceError("cache_identity_or_file_mismatch")
        body = source.read_bytes()
        if hashlib.sha256(body).hexdigest() != value.get("sha256") or len(body) != value.get("byte_count"):
            raise SourceError("source_checksum_mismatch")
        return True

    def save(self, task, text, identity):
        source, metadata = self.paths(task)
        if source.exists() or metadata.exists():
            raise SourceError("refusing_source_overwrite")
        body = text.encode("utf-8")
        # Preserve DOM text exactly, including all whitespace. It is not a
        # claim to have recovered the original uploaded file's bytes.
        with source.open("xb") as stream:
            stream.write(body)
        write_json(metadata, {"version": 1, "task": task, "url": source_url(task),
                              "retrieved_at_utc": utc_now(), "extraction": "DOM textContent, UTF-8",
                              "selector": SELECTOR, "sha256": hashlib.sha256(body).hexdigest(),
                              "byte_count": len(body), "label": "unknown", **identity})

    def event(self, value):
        with (self.root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"at_utc": utc_now(), **value}) + "\n")


class MemorySampler:
    """Process-tree RSS sum includes shared pages more than once; not PSS."""
    def __init__(self, min_free_mb, max_rss_mb):
        import psutil
        self.psutil = psutil
        self.root = psutil.Process()
        self.min_free_mb, self.max_rss_mb = min_free_mb, max_rss_mb
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples = 0
        self.peak_rss_mb = 0
        self.min_available_mb = None
        self.failure = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        rss = 0
        try:
            for process in [self.root] + self.root.children(recursive=True):
                try:
                    rss += process.memory_info().rss
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    continue
            available = self.psutil.virtual_memory().available / 1024 ** 2
            rss /= 1024 ** 2
            with self.lock:
                self.samples += 1
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
                self.min_available_mb = available if self.min_available_mb is None else min(self.min_available_mb, available)
                if (self.min_free_mb and available < self.min_free_mb) or (self.max_rss_mb and rss > self.max_rss_mb):
                    self.failure = "memory_budget_exceeded"
        except self.psutil.Error:
            with self.lock:
                self.failure = "memory_measurement_unavailable"

    def _loop(self):
        while not self.stop.wait(.5):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def check(self):
        if self.failure:
            raise SourceError(self.failure)

    def report(self):
        return {"samples": self.samples, "peak_process_tree_rss_sum_mb": round(self.peak_rss_mb, 1),
                "minimum_system_available_mb": round(self.min_available_mb, 1) if self.min_available_mb is not None else None,
                "metric": "collector + descendants RSS sum; shared pages may be counted multiple times; sampled every 0.5s"}

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)


def collect_sources(config, archive, fetch, memory, limit, clock=time.monotonic, sleep=time.sleep):
    """Testable acquisition loop. Zero requests for a fully cached selection."""
    if limit < 1:
        raise ValueError("Limit must be positive")
    tasks = config["submissions"][:limit]
    result = {"run_id": config["run_id"], "selected": len(tasks), "downloaded": 0,
              "cache_hits": 0, "network_navigations": 0, "status": "incomplete", "error": None}
    last_start = None
    try:
        for task in tasks:
            memory.check()
            if archive.cached(task):
                result["cache_hits"] += 1
                continue
            if last_start is not None:
                wait = config["request_interval_seconds"] - (clock() - last_start)
                if wait > 0:
                    sleep(wait)
            memory.check()
            last_start = clock()
            result["network_navigations"] += 1
            try:
                final_url, status, snapshot = fetch(task)
                source, identity = validate_snapshot(task, final_url, status, snapshot)
                memory.check()
                archive.save(task, source, identity)
                archive.event({"submission_id": task["submission_id"], "status": "saved", "http_status": status})
                result["downloaded"] += 1
                print(f"Saved submission {task['submission_id']}", flush=True)
            except SourceError as error:
                archive.event({"submission_id": task["submission_id"], "status": "stopped", "reason": str(error)})
                raise
        result["status"] = "selected_batch_complete"
    except SourceError as error:
        result["error"] = str(error)
    try:
        result["entire_manifest_complete"] = all(archive.cached(task) for task in config["submissions"])
    except SourceError as error:
        result["entire_manifest_complete"] = False
        result["status"], result["error"] = "incomplete", str(error)
    return result


def browser_context(playwright, profile, channel, headed):
    context = playwright.chromium.launch_persistent_context(
        str(profile), channel=channel, headless=not headed,
        accept_downloads=False, chromium_sandbox=True,
        viewport={"width": 1000, "height": 750})
    # The dedicated profile contains no unrelated tabs.
    page = context.pages[0] if context.pages else context.new_page()
    for other in context.pages[1:]:
        other.close()
    page.on("dialog", lambda dialog: dialog.dismiss())
    context.on("page", lambda popup: popup.close())
    return context, page


def login(profile, channel, seconds):
    from playwright.sync_api import sync_playwright, Error
    profile.mkdir(parents=True, exist_ok=True)
    with exclusive_file(profile / ".collector-lock"), sync_playwright() as playwright:
        try:
            context, page = browser_context(playwright, profile, channel, True)
        except Error:
            raise SourceError("browser_startup_failed; run login from a normal local PowerShell window") from None
        try:
            try:
                page.goto("https://codeforces.com/enter", wait_until="domcontentloaded", timeout=30000)
            except Error:
                print("Login page did not finish loading. Use the visible browser to sign in.", flush=True)
            print("Sign into Codeforces in the separate browser window. No password is read by this script.", flush=True)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if page.is_closed():
                    raise SourceError("login_window_closed_before_detection")
                if urlsplit(page.url).netloc == "codeforces.com" and page.locator('a[href*="/logout"]').count():
                    write_json(profile / "login-ready.json", {"at_utc": utc_now(), "channel": channel})
                    print("Login detected and profile saved. Ready for source collection.", flush=True)
                    return 0
                page.wait_for_timeout(1000)
            raise SourceError("login_wait_expired")
        finally:
            context.close()


def run(config, profile, channel, headed, limit):
    from playwright.sync_api import sync_playwright, Error
    profile.mkdir(parents=True, exist_ok=True)
    archive = Archive(PROJECT / "data" / "sources" / config["run_id"], config)
    with exclusive_file(profile / ".collector-lock"), MemorySampler(config["min_free_mb"], config["max_process_rss_mb"]) as memory:
        context = None
        page = None
        with sync_playwright() as playwright:
            def fetch(task):
                nonlocal context, page
                try:
                    if context is None:
                        context, page = browser_context(playwright, profile, channel, headed)
                    response = page.goto(source_url(task), wait_until="domcontentloaded", timeout=config["timeout_seconds"] * 1000)
                    status = response.status if response else None
                    if status in (401, 403, 429):
                        raise SourceError("access_challenge_or_denial")
                    # A bounded wait for source rendering. Missing/denied sources never
                    # become empty successful records. No challenge solver or stealth.
                    if page.locator(SELECTOR).count() == 0:
                        try:
                            page.locator(SELECTOR).wait_for(state="attached", timeout=5000)
                        except Error:
                            pass
                    return page.url, status, page.evaluate(SNAPSHOT_JS)
                except SourceError:
                    raise
                except Error:
                    raise SourceError("browser_navigation_or_extraction_failed") from None
            try:
                result = collect_sources(config, archive, fetch, memory, limit)
            finally:
                if context is not None:
                    context.close()
        result.update(memory.report())
        result.update({"browser_channel": channel, "headed": headed, "at_utc": utc_now()})
        write_json(archive.root / "last_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == "selected_batch_complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    signin = commands.add_parser("login", help="Sign in manually once in a separate local profile")
    signin.add_argument("--seconds", type=int, default=600)
    download = commands.add_parser("collect", help="Collect a bounded batch of specified source pages")
    download.add_argument("--manifest", type=Path, default=PROJECT / "config" / "source-feasibility.json")
    download.add_argument("--limit", type=int, default=5)
    download.add_argument("--headed", action="store_true")
    for subparser in (signin, download):
        subparser.add_argument("--channel", choices=["msedge", "chrome", "chromium"], default="msedge")
    args = parser.parse_args(argv)
    try:
        if args.command == "login":
            if not 30 <= args.seconds <= 1800:
                raise ValueError("Login timeout must be between 30 and 1800 seconds")
            return login(PROFILE, args.channel, args.seconds)
        return run(read_config(args.manifest), PROFILE, args.channel, args.headed, args.limit)
    except (SourceError, ValueError, OSError, ImportError) as error:
        print(f"Source collector stopped: {error}", flush=True)
        return 2
    except KeyboardInterrupt:
        print("Source collector interrupted; completed artifacts are retained.", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
