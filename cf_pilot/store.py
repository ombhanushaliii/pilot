"""SQLite checkpoints and immutable, checksummed response archives."""

from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, created_at REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'collecting', error TEXT
);
CREATE TABLE IF NOT EXISTS responses (
 run_id TEXT NOT NULL, request_key TEXT NOT NULL, url TEXT NOT NULL,
 params_json TEXT NOT NULL, raw_path TEXT NOT NULL, sha256 TEXT NOT NULL,
 fetched_at REAL NOT NULL, byte_count INTEGER NOT NULL,
 PRIMARY KEY(run_id, request_key)
);
CREATE TABLE IF NOT EXISTS request_attempts (
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, request_key TEXT NOT NULL,
 started_at REAL NOT NULL, outcome TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS rate_gate (id INTEGER PRIMARY KEY CHECK(id=1), last_start REAL NOT NULL);
INSERT OR IGNORE INTO rate_gate VALUES(1, 0);
CREATE TABLE IF NOT EXISTS jobs (
 run_id TEXT NOT NULL, job_key TEXT NOT NULL, next_from INTEGER NOT NULL DEFAULT 1,
 status TEXT NOT NULL DEFAULT 'pending', pages INTEGER NOT NULL DEFAULT 0,
 detail TEXT, PRIMARY KEY(run_id, job_key)
);
CREATE TABLE IF NOT EXISTS cohort (
 run_id TEXT NOT NULL, handle TEXT NOT NULL COLLATE NOCASE, current_rating INTEGER,
 selected INTEGER NOT NULL, source_url TEXT NOT NULL, PRIMARY KEY(run_id, handle)
);
CREATE TABLE IF NOT EXISTS contests (
 run_id TEXT NOT NULL, contest_id INTEGER NOT NULL, payload_json TEXT NOT NULL,
 selected INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(run_id, contest_id)
);
CREATE TABLE IF NOT EXISTS rating_events (
 run_id TEXT NOT NULL, handle TEXT NOT NULL COLLATE NOCASE, contest_id INTEGER NOT NULL,
 payload_json TEXT NOT NULL, PRIMARY KEY(run_id, handle, contest_id)
);
CREATE TABLE IF NOT EXISTS submissions (
 run_id TEXT NOT NULL, submission_id INTEGER NOT NULL, handle TEXT COLLATE NOCASE,
 contest_id INTEGER, created_at INTEGER NOT NULL, payload_json TEXT NOT NULL,
 PRIMARY KEY(run_id, submission_id)
);
CREATE TABLE IF NOT EXISTS submission_sources (
 run_id TEXT NOT NULL, submission_id INTEGER NOT NULL, request_key TEXT NOT NULL,
 PRIMARY KEY(run_id, submission_id, request_key)
);
CREATE TABLE IF NOT EXISTS standings (
 run_id TEXT NOT NULL, contest_id INTEGER NOT NULL, handle TEXT NOT NULL COLLATE NOCASE,
 payload_json TEXT NOT NULL, PRIMARY KEY(run_id, contest_id, handle)
);
CREATE TABLE IF NOT EXISTS submission_conflicts (
 run_id TEXT NOT NULL, submission_id INTEGER NOT NULL, request_key TEXT NOT NULL,
 detail_json TEXT NOT NULL, PRIMARY KEY(run_id, submission_id, request_key)
);
CREATE TABLE IF NOT EXISTS sessions (
 run_id TEXT NOT NULL, contest_id INTEGER NOT NULL, handle TEXT NOT NULL COLLATE NOCASE,
 eligible INTEGER NOT NULL, exclusion_reason TEXT, submission_count INTEGER NOT NULL,
 prior_contest_count INTEGER NOT NULL, history_complete INTEGER NOT NULL,
 in_standings INTEGER NOT NULL, PRIMARY KEY(run_id, contest_id, handle)
);
CREATE TABLE IF NOT EXISTS labels (
 run_id TEXT NOT NULL, contest_id INTEGER NOT NULL, handle TEXT NOT NULL COLLATE NOCASE,
 label TEXT NOT NULL DEFAULT 'unknown', evidence_url TEXT, evidence_type TEXT,
 confidence TEXT, reviewer_notes TEXT, PRIMARY KEY(run_id, contest_id, handle)
);
CREATE INDEX IF NOT EXISTS submission_lookup ON submissions(run_id, handle, created_at);
"""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "pilot.sqlite", timeout=90)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    def start_run(self, run_id: str, config: dict):
        encoded = canonical(config)
        row = self.db.execute("SELECT config_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row and row[0] != encoded:
            raise ValueError("This run_id already has different settings. Use a new run_id.")
        self.db.execute("INSERT OR IGNORE INTO runs(run_id,config_json,created_at) VALUES(?,?,?)",
                        (run_id, encoded, time.time()))
        self.db.execute("UPDATE runs SET status='collecting',error=NULL WHERE run_id=?", (run_id,))
        self.db.commit()

    def archive(self, run_id, key, url, params, body):
        digest = hashlib.sha256(body).hexdigest()
        path = self.root / "raw" / run_id / (key + ".gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        # Never replace an existing archived response for the same run.
        existing = self.db.execute("SELECT sha256 FROM responses WHERE run_id=? AND request_key=?",
                                   (run_id, key)).fetchone()
        if existing:
            if existing[0] != digest:
                raise ValueError("Conflicting response for a frozen request")
            return
        temp.write_bytes(gzip.compress(body, mtime=0))
        temp.replace(path)
        self.db.execute("INSERT INTO responses VALUES(?,?,?,?,?,?,?,?)",
                        (run_id, key, url, canonical(params), str(path.relative_to(self.root)),
                         digest, time.time(), len(body)))
        self.db.commit()

    def cached(self, run_id, key):
        row = self.db.execute("SELECT raw_path,sha256 FROM responses WHERE run_id=? AND request_key=?",
                              (run_id, key)).fetchone()
        if not row:
            return None
        try:
            body = gzip.decompress((self.root / row[0]).read_bytes())
        except (OSError, EOFError) as e:
            raise ValueError(f"Archived response unreadable: {row[0]}") from e
        if hashlib.sha256(body).hexdigest() != row[1]:
            raise ValueError(f"Archived response checksum mismatch: {row[0]}")
        return body

    @contextmanager
    def request_slot(self, interval, clock=time.time, sleep=time.sleep):
        """Serialize starts across processes sharing this DB, including the network call."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            last = self.db.execute("SELECT last_start FROM rate_gate WHERE id=1").fetchone()[0]
            delay = max(0.0, last + interval - clock())
            if delay:
                sleep(delay)
            started = clock()
            self.db.execute("UPDATE rate_gate SET last_start=? WHERE id=1", (started,))
            yield started
        finally:
            # Even failed requests consume a slot.
            self.db.commit()

    def job(self, run_id, key):
        self.db.execute("INSERT OR IGNORE INTO jobs(run_id,job_key) VALUES(?,?)", (run_id, key))
        self.db.commit()
        return self.db.execute("SELECT * FROM jobs WHERE run_id=? AND job_key=?", (run_id, key)).fetchone()

    def update_job(self, run_id, key, next_from, status, pages, detail=None):
        self.db.execute("UPDATE jobs SET next_from=?,status=?,pages=?,detail=? WHERE run_id=? AND job_key=?",
                        (next_from, status, pages, detail, run_id, key))
        self.db.commit()

    def ingest_submissions(self, run_id, key, items):
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                raise ValueError("Malformed submission ID")
            created = item.get("creationTimeSeconds")
            if not isinstance(created, int):
                raise ValueError("Malformed submission timestamp")
            members = item.get("author", {}).get("members", [])
            handle = members[0].get("handle") if len(members) == 1 else None
            existing = self.db.execute("SELECT payload_json FROM submissions WHERE run_id=? AND submission_id=?",
                                       (run_id, item["id"])).fetchone()
            if existing:
                original = json.loads(existing[0])
                fields = ["contestId", "creationTimeSeconds", "verdict", "testset", "passedTestCount",
                          "programmingLanguage", "points", "timeConsumedMillis", "memoryConsumedBytes"]
                changes = {field: [original.get(field), item.get(field)] for field in fields
                           if original.get(field) != item.get(field)}
                if changes:
                    self.db.execute("INSERT OR IGNORE INTO submission_conflicts VALUES(?,?,?,?)",
                                    (run_id, item["id"], key, canonical(changes)))
            self.db.execute("INSERT OR IGNORE INTO submissions VALUES(?,?,?,?,?,?)",
                            (run_id, item["id"], handle, item.get("contestId"), created, canonical(item)))
            self.db.execute("INSERT OR IGNORE INTO submission_sources VALUES(?,?,?)",
                            (run_id, item["id"], key))
        self.db.commit()
