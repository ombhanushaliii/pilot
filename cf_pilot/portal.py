"""Local post-contest review portal backed by frozen data and actual RL inference."""
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time
from urllib.parse import parse_qs, unquote, urlsplit

import torch

from .benchmark import FEATURE_GROUPS, behavior_features, file_hash
from .deep_rl import QNetwork, infer
from .outcomes import audit_outcomes
from .rl_data import load_data, binary_metrics


ACTIONS = ("inspect_pacing", "inspect_trajectory", "inspect_language", "low_priority", "review")


class Portal:
    def __init__(self, root):
        self.root = Path(root)
        torch.set_num_threads(1)
        config = json.loads((self.root / "config/deep-rl.json").read_text())
        self.models = self.root / "data" / config["run_id"]
        self.folder = self.root / "data" / config["benchmark_id"]
        self.selection = json.loads((self.models / "selection.json").read_text())
        complete = json.loads((self.models / "complete.json").read_text())
        for path in (self.models / "selection.json", self.models / (self.selection["selected"] + ".pt")):
            if file_hash(path) != complete["artifacts"][str(path.relative_to(self.root))]:
                raise ValueError("Frozen policy artifact changed")
        candidate = self.selection["selected_candidate"]
        self.network = QNetwork(candidate["width"], candidate["dueling"])
        self.network.load_state_dict(torch.load(self.models / (self.selection["selected"] + ".pt"), map_location="cpu", weights_only=True))
        self.network.eval()
        summary = json.loads((self.folder / "summary.json").read_text())
        pre = json.loads((self.folder / "preprocessor.json").read_text())
        self.data = load_data(self.folder, "test", summary, pre)
        manifest = json.loads((self.folder / "manifest.json").read_text())
        ids = {int(m["contest_id"]) for m in self.data["metadata"]}
        if len(ids) != 1:
            raise ValueError("Demo requires exactly one frozen target contest")
        self.contest_id = ids.pop()
        self.contest = manifest["source_contests"][str(self.contest_id)]["contest"]
        self.rows, self.by_id = [], {}
        with (self.folder / "test/features.csv").open(encoding="utf-8", newline="") as stream:
            for i, feature in enumerate(csv.DictReader(stream)):
                meta = self.data["metadata"][i]
                if feature["session_id"] != self.data["keys"][i]:
                    raise ValueError("Portal feature join differs from frozen model inputs")
                raw = {name: float(value) if value else None for name, value in feature.items() if name != "session_id"}
                row = {"id": feature["session_id"], "handle": meta["handle"], "attempts": int(raw["attempts"]),
                       "problem_count": int(raw["attempted_problems"]), "last_seconds": round(raw["last_submission_fraction"] * self.contest["durationSeconds"]),
                       "features": raw, "submissions": [], "status": "unchecked", "actions": [], "inspections": None, "model_index": i}
                self.rows.append(row)
                self.by_id[row["id"]] = row
        source = self.root / "data/rl-expansion-2026-002/pilot.sqlite"
        if file_hash(source) != manifest["source_fingerprints"]["rl-expansion-2026-002"]:
            raise ValueError("Raw submission snapshot changed")
        handles = {r["handle"].casefold(): r for r in self.rows}
        def cache_session(handle, items):
            row = handles.get(handle)
            if row is None:
                return
            computed = behavior_features(items, self.contest)
            if computed["attempts"] != row["attempts"] or computed["attempted_problems"] != row["problem_count"]:
                raise ValueError("Displayed submissions disagree with frozen session coverage")
            row["submissions"] = [{"id": item["id"], "problem": item["problem"]["index"], "problem_name": item["problem"].get("name", ""),
                "language": item.get("programmingLanguage") or "Unknown", "verdict": item.get("verdict") or "UNAVAILABLE",
                "seconds": item["creationTimeSeconds"] - self.contest["startTimeSeconds"],
                "url": f"https://codeforces.com/contest/{self.contest_id}/submission/{item['id']}"} for item in sorted(items, key=lambda s: (s["creationTimeSeconds"], s["id"]))]
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
            self.snapshot_time = db.execute("SELECT max(fetched_at) FROM responses").fetchone()[0]
            payloads = (json.loads(r[0]) for r in db.execute("SELECT payload_json FROM submissions WHERE contest_id=? ORDER BY submission_id", (self.contest_id,)))
            self.audit = audit_outcomes(payloads, self.contest, True, year=2026, on_session=cache_session)
        if any(len(r["submissions"]) != r["attempts"] for r in self.rows):
            raise ValueError("Missing submission details for a frozen session")
        self.expected = json.loads((self.root / "reports" / config["run_id"] / "results.json").read_text())["fresh_test"][self.selection["selected"]]["all"]
        self.lock = Lock()
        self.reviewed = False
        self.elapsed = None
        self.review_runs = 0

    def overview(self):
        return {"contest_id": self.contest_id, "name": self.contest["name"], "start_time": self.contest["startTimeSeconds"],
                "duration": self.contest["durationSeconds"], "snapshot_time": self.snapshot_time,
                "participants": len(self.rows), "submissions": sum(r["attempts"] for r in self.rows),
                "reviewed": self.reviewed, "flagged": sum(r["status"] == "review" for r in self.rows) if self.reviewed else None,
                "elapsed_seconds": self.elapsed, "model": self.selection["selected"], "algorithm": "Dueling Double DQN",
                "checkpoint": self.selection["selected_update"], "metrics": self.expected if self.reviewed else None,
                "scope": "Frozen benchmark-eligible sessions; not an official ranklist or an untouched test",
                "interpretation": "Review priority is not a misconduct finding"}

    @staticmethod
    def public_row(row):
        return {key: row[key] for key in ("id", "handle", "attempts", "problem_count", "last_seconds", "status", "inspections")}

    def participants(self, *, query="", status="all", page=1, page_size=40):
        if status not in {"all", "review", "low_priority"} or not 1 <= page_size <= 100 or page < 1:
            raise ValueError("Invalid participant filter or pagination")
        matched = [r for r in self.rows if query.casefold() in r["handle"].casefold() and (status == "all" or r["status"] == status)]
        matched.sort(key=lambda r: (r["status"] != "review", r["handle"].casefold()))
        offset = (page - 1) * page_size
        return {"rows": [self.public_row(r) for r in matched[offset:offset + page_size]], "total": len(matched), "page": page, "page_size": page_size}

    def detail(self, session_id):
        row = self.by_id[session_id]
        return self.public_row(row) | {"submissions": row["submissions"], "actions": row["actions"],
            "evidence": {group: {name: row["features"][name] for name in columns} for group, columns in FEATURE_GROUPS.items()},
            "duration": self.contest["durationSeconds"]}

    def run_review(self):
        with self.lock:
            if self.reviewed:
                return self.overview() | {"cached": True}
            started = time.perf_counter()
            # Only the numeric observation matrix enters the policy. Targets remain outside inference.
            prediction, counts, _, traces = infer(self.network, self.data["x"], traces=True)
            measured = binary_metrics(self.data["y"], prediction, counts)
            if measured != self.expected:
                raise ValueError("Restored policy does not reproduce the frozen evaluation")
            for row in self.rows:
                idx = row["model_index"]
                row["status"] = "review" if prediction[idx] else "low_priority"
                row["inspections"] = int(counts[idx])
                row["actions"] = [ACTIONS[a] for a in traces[idx]]
            self.elapsed = round(time.perf_counter() - started, 3)
            self.review_runs += 1
            self.reviewed = True
            return self.overview() | {"cached": False}


def handler_for(portal, ui_root):
    ui_root = Path(ui_root)
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            try:
                host = urlsplit("http://" + self.headers.get("Host", ""))
                return host.hostname in {"127.0.0.1", "localhost"} and host.port == self.server.server_port
            except ValueError:
                return False

        def reply(self, status, value, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.local_host():
                return self.reply(403, {"error": "This workspace is available on localhost only"})
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/contest":
                    return self.reply(200, portal.overview())
                if parsed.path == "/api/participants":
                    args = parse_qs(parsed.query)
                    result = portal.participants(query=args.get("q", [""])[0][:100], status=args.get("status", ["all"])[0],
                        page=int(args.get("page", [1])[0]), page_size=int(args.get("page_size", [40])[0]))
                    return self.reply(200, result)
                if parsed.path.startswith("/api/participants/"):
                    return self.reply(200, portal.detail(unquote(parsed.path.removeprefix("/api/participants/"))))
                files = {"/": ("index.html", "text/html; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
                if parsed.path in files:
                    name, content_type = files[parsed.path]
                    return self.reply(200, (ui_root / name).read_bytes(), content_type)
                return self.reply(404, {"error": "Page not found"})
            except KeyError:
                return self.reply(404, {"error": "Participant not found"})
            except ValueError as error:
                return self.reply(400, {"error": str(error)})

        def do_POST(self):
            if not self.local_host():
                return self.reply(403, {"error": "This workspace is available on localhost only"})
            origin = self.headers.get("Origin")
            expected = "http://" + self.headers.get("Host", "")
            if origin and origin != expected:
                return self.reply(403, {"error": "Review requests must originate from this local portal"})
            if self.path != "/api/review":
                return self.reply(404, {"error": "Action not found"})
            try:
                return self.reply(200, portal.run_review())
            except (ValueError, RuntimeError) as error:
                return self.reply(500, {"error": str(error)})
    return Handler


def serve(root, port=8765):
    portal = Portal(root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(portal, Path(root) / "ui"))
    print(f"Pilot review portal: http://127.0.0.1:{port}", flush=True)
    print(f"Loaded {len(portal.rows):,} frozen sessions; saved model ready. No collection or training.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
