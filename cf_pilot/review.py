"""Append-only evidence reviews with atomic, explicit label application."""

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .sessions import eligibility
from .store import canonical


REVIEW_VERSION = "session_evidence_v1"
LABELS = {"unknown", "presumed_negative", "substantiated_misconduct", "substantiated_ai_misuse"}
POSITIVE_TYPES = {"platform_finding", "organizer_finding", "verified_admission"}
COLUMNS = (
    "run_id", "contest_id", "handle", "label", "evidence_url", "evidence_type",
    "source_collected_at", "reviewed_at", "reviewer", "second_reviewer", "second_decision",
    "confidence", "session_verified", "ai_explicit", "submission_ids", "related_case_id",
    "sampling_reason", "disagreement", "reversal", "reviewer_notes",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_reviews (
 review_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, contest_id INTEGER NOT NULL,
 handle TEXT NOT NULL COLLATE NOCASE, payload_json TEXT NOT NULL, row_sha256 TEXT NOT NULL,
 imported_at TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 UNIQUE(run_id,contest_id,handle,row_sha256)
);
"""


def initialize_review(store):
    store.db.executescript(SCHEMA)
    store.db.commit()


def export_review_register(store, run_id, path, limit=100, seed=736):
    """Reproducible random feasibility batch from selected eligible sessions."""
    if limit < 1:
        raise ValueError("Review limit must be positive")
    rows = list(store.db.execute(
        "SELECT s.run_id,s.contest_id,s.handle,l.label FROM sessions s "
        "JOIN labels l USING(run_id,contest_id,handle) "
        "JOIN participant_coverage p USING(run_id,contest_id,handle) "
        "WHERE s.run_id=? AND s.eligible=1 AND p.selected=1", (run_id,)))
    if not rows:
        raise ValueError("No eligible development sessions for review")
    rows.sort(key=lambda x: hashlib.sha256(
        f"{seed}:{x['run_id']}:{x['contest_id']}:{x['handle'].casefold()}".encode()).hexdigest())
    path = Path(path)
    if path.exists():
        return {"path": str(path), "preserved_existing": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for entry in rows[:limit]:
            # New worksheets contain pending unknown decisions. Existing DB outcomes
            # are not copied without their evidence; untouched rows cannot overwrite them.
            writer.writerow({**dict(entry), "label": "unknown", "sampling_reason": "random_development_sample",
                             "disagreement": "no", "reversal": "no"})
    return {"path": str(path), "rows": min(limit, len(rows)), "seed": seed,
            "sampling_note": "Random within weighted development sample, not representative of the full population.",
            "preserved_existing": False}


def validate_date(value, field):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        if parsed > datetime.now(timezone.utc):
            raise ValueError("date is in the future")
        return parsed
    except (ValueError, TypeError) as error:
        raise ValueError(f"{field} requires a non-future ISO 8601 timestamp with timezone") from error


def export_review_context(store, run_id, register, output):
    """Evidence navigation for the batch, without a suspicion score or decision."""
    with Path(register).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Review context requires the versioned register")
        rows = list(reader)
    cases = []
    for row in rows:
        if row["run_id"] != run_id:
            raise ValueError("Wrong run_id in review context")
        contest_id = int(row["contest_id"])
        session = store.db.execute("SELECT * FROM sessions WHERE run_id=? AND contest_id=? AND handle=?",
                                   (run_id, contest_id, row["handle"])).fetchone()
        if not session or not session["eligible"]:
            raise ValueError("Review context requires eligible sessions")
        contest = json.loads(store.db.execute("SELECT payload_json FROM contests WHERE run_id=? AND contest_id=?",
                                            (run_id, contest_id)).fetchone()[0])
        items = [json.loads(x[0]) for x in store.db.execute("SELECT payload_json FROM submissions "
            "WHERE run_id=? AND contest_id=? AND handle=? ORDER BY created_at,submission_id",
            (run_id, contest_id, row["handle"]))]
        items = [x for x in items if eligibility(x, contest)[0]]
        cases.append({"run_id": run_id, "contest_id": contest_id, "handle": session["handle"],
                      "contest_url": f"https://codeforces.com/contest/{contest_id}",
                      "official_submission_count": len(items),
                      "observed_prior_contests": session["prior_contest_count"],
                      "submissions": [{"id": x["id"], "creation_time_seconds": x["creationTimeSeconds"],
                                       "offset_seconds": x["creationTimeSeconds"] - contest["startTimeSeconds"],
                                       "problem_index": x["problem"]["index"],
                                       "submission_url": f"https://codeforces.com/contest/{contest_id}/submission/{x['id']}"}
                                      for x in items]})
    body = (json.dumps({"run_id": run_id, "cases": cases,
        "note": "Navigation from archived submission IDs. Linked pages have not been fetched or verified here. "
                "Timing is context, not independent misconduct evidence; no score or label is assigned."},
        indent=2, ensure_ascii=False) + "\n").encode()
    output = Path(output)
    if output.exists() and output.read_bytes() != body:
        raise ValueError("Existing review context differs; use a new destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(body)
    return {"path": str(output), "cases": len(cases)}


def validate_row(store, row, run_id):
    row = {key: (value or "").strip() for key, value in row.items()}
    if row["run_id"] != run_id:
        raise ValueError("Wrong run_id")
    try:
        contest_id = int(row["contest_id"])
    except ValueError as error:
        raise ValueError("Invalid contest_id") from error
    session = store.db.execute("SELECT eligible FROM sessions WHERE run_id=? AND contest_id=? AND handle=?",
                               (run_id, contest_id, row["handle"])).fetchone()
    if not session or not session[0]:
        raise ValueError("Label requires an existing eligible session")
    if not store.db.execute("SELECT 1 FROM labels WHERE run_id=? AND contest_id=? AND handle=?",
                            (run_id, contest_id, row["handle"])).fetchone():
        raise ValueError("Session has no existing label record")
    if row["label"] not in LABELS:
        raise ValueError("Unsupported label category")
    if row["confidence"] not in {"", "low", "medium", "high"}:
        raise ValueError("Confidence must be blank, low, medium or high")
    for name in ("disagreement", "reversal"):
        if row[name] not in {"yes", "no"}:
            raise ValueError(f"{name} must be yes or no")
    for name in ("session_verified", "ai_explicit"):
        if row[name] not in {"", "yes", "no"}:
            raise ValueError(f"{name} must be blank, yes or no")
    if row["second_decision"] and (not row["second_reviewer"] or row["second_decision"] not in LABELS):
        raise ValueError("Second decision requires a reviewer and valid label")
    if row["second_reviewer"] and not row["second_decision"]:
        raise ValueError("Second reviewer requires a decision")
    if row["second_reviewer"] and row["second_reviewer"].casefold() == row["reviewer"].casefold():
        raise ValueError("Second reviewer must be distinct")
    if (row["disagreement"] == "yes" or row["reversal"] == "yes"
            or (row["second_decision"] and row["second_decision"] != row["label"])) and row["label"] != "unknown":
        raise ValueError("Disputed or reversed cases must remain unknown")
    if row["evidence_url"]:
        url = urlsplit(row["evidence_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Evidence URL must be a public HTTP(S) URL without credentials")
    ids = []
    if row["submission_ids"]:
        try:
            ids = [int(x.strip()) for x in row["submission_ids"].split(";")]
        except ValueError as error:
            raise ValueError("submission_ids must be semicolon-separated integers") from error
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence submission IDs")
        for identifier in ids:
            item = store.db.execute("SELECT handle,contest_id FROM submissions WHERE run_id=? AND submission_id=?",
                                     (run_id, identifier)).fetchone()
            if not item or item[0].casefold() != row["handle"].casefold() or item[1] != contest_id:
                raise ValueError("Evidence submission does not belong to this session")
    meaningful = (row["label"] != "unknown" or row["disagreement"] == "yes" or row["reversal"] == "yes"
                  or any(row[name] for name in (
        "evidence_url", "evidence_type", "reviewer", "reviewer_notes", "reviewed_at", "source_collected_at",
        "second_reviewer", "second_decision", "confidence", "submission_ids", "related_case_id")))
    if meaningful:
        if not row["reviewer"] or not row["reviewer_notes"]:
            raise ValueError("Reviewed entries require a reviewer and assessment notes")
        reviewed = validate_date(row["reviewed_at"], "reviewed_at")
        contest = json.loads(store.db.execute("SELECT payload_json FROM contests WHERE run_id=? AND contest_id=?",
                                            (run_id, contest_id)).fetchone()[0])
        if reviewed.timestamp() < contest["startTimeSeconds"] + contest["durationSeconds"]:
            raise ValueError("Review must follow the completed contest")
        if row["source_collected_at"]:
            collected = validate_date(row["source_collected_at"], "source_collected_at")
            if collected.timestamp() < contest["startTimeSeconds"] + contest["durationSeconds"]:
                raise ValueError("Source collection must follow the completed contest")
            if collected > reviewed:
                raise ValueError("Source collection must not follow review")
    if row["label"] != "unknown":
        if row["confidence"] not in {"medium", "high"} or row["session_verified"] != "yes":
            raise ValueError("Decided labels require session verification and medium/high confidence")
    if row["label"].startswith("substantiated_"):
        if not row["evidence_url"] or row["evidence_type"] not in POSITIVE_TYPES or not row["source_collected_at"]:
            raise ValueError("Positive label requires independent finding/admission source and collection date")
        if not row["related_case_id"]:
            raise ValueError("Positive label requires a case/group ID for later split integrity")
        if row["label"] == "substantiated_ai_misuse" and row["ai_explicit"] != "yes":
            raise ValueError("AI misuse requires explicit AI-specific evidence")
    if row["label"] == "presumed_negative" and row["evidence_type"] != "reviewed_no_independent_finding":
        raise ValueError("Presumed negative requires explicit review category; it is not verified fair")
    row["contest_id"] = str(contest_id)
    # Normalize handles to the stored spelling to make imports idempotent.
    row["handle"] = store.db.execute("SELECT handle FROM sessions WHERE run_id=? AND contest_id=? AND handle=?",
                                    (run_id, contest_id, row["handle"])).fetchone()[0]
    return row, meaningful


def import_reviews(store, run_id, path, apply=False):
    """Validate all rows before mutation. Untouched unknown template rows are no-ops."""
    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Review register header does not match session_evidence_v1")
        incoming = list(reader)
    checked = []
    errors = []
    seen = set()
    for number, row in enumerate(incoming, 2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row length")
            valid, meaningful = validate_row(store, row, run_id)
            key = (valid["contest_id"], valid["handle"].casefold())
            if key in seen:
                raise ValueError("Duplicate session in import")
            seen.add(key)
            if meaningful:
                checked.append(valid)
        except (ValueError, AttributeError) as error:
            errors.append({"row": number, "error": str(error)})
    if errors:
        raise ValueError("Review import rejected without changes: " + json.dumps(errors))
    result = {"validated_rows": len(incoming), "reviewed_rows": len(checked),
              "untouched_unknown_rows": len(incoming) - len(checked),
              "labels": dict(Counter(x["label"] for x in checked)), "applied": bool(apply),
              "source_sha256": source_hash,
              "verification_limit": "Structural validation only. Reviewers must independently inspect source evidence."}
    if apply:
        initialize_review(store)
        changes = 0
        with store.db:
            for row in checked:
                payload = canonical(row)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                cursor = store.db.execute("INSERT OR IGNORE INTO evidence_reviews "
                    "(run_id,contest_id,handle,payload_json,row_sha256,imported_at,source_sha256) VALUES(?,?,?,?,?,?,?)",
                    (run_id, int(row["contest_id"]), row["handle"], payload, digest,
                     datetime.now(timezone.utc).isoformat(), source_hash))
                if cursor.rowcount:
                    changes += 1
                    store.db.execute("UPDATE labels SET label=?,evidence_url=?,evidence_type=?,confidence=?,reviewer_notes=? "
                        "WHERE run_id=? AND contest_id=? AND handle=?",
                        (row["label"], row["evidence_url"], row["evidence_type"], row["confidence"],
                         row["reviewer_notes"], run_id, int(row["contest_id"]), row["handle"]))
        result["new_review_events"] = changes
    return result


def review_audit(store, run_id):
    if not store.db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
        raise ValueError("Unknown run_id")
    counts = dict(Counter(row[0] for row in store.db.execute("SELECT l.label FROM labels l JOIN sessions s "
        "USING(run_id,contest_id,handle) WHERE l.run_id=? AND s.eligible=1", (run_id,))))
    has_table = store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_reviews'").fetchone()
    events = store.db.execute("SELECT count(*) FROM evidence_reviews WHERE run_id=?", (run_id,)).fetchone()[0] if has_table else 0
    reviewed_sessions = store.db.execute("SELECT count(*) FROM (SELECT contest_id,handle FROM evidence_reviews "
        "WHERE run_id=? GROUP BY contest_id,handle)", (run_id,)).fetchone()[0] if has_table else 0
    issues = []
    for row in store.db.execute("SELECT l.* FROM labels l JOIN sessions s USING(run_id,contest_id,handle) "
                               "WHERE l.run_id=? AND s.eligible=1 AND l.label!='unknown'", (run_id,)):
        latest = store.db.execute("SELECT payload_json FROM evidence_reviews WHERE run_id=? AND contest_id=? "
            "AND handle=? ORDER BY review_id DESC LIMIT 1", (run_id, row["contest_id"], row["handle"])).fetchone() if has_table else None
        if not latest or any(json.loads(latest[0])[name] != (row[name] or "") for name in
                             ("label", "evidence_url", "evidence_type", "confidence", "reviewer_notes")):
            issues.append({"contest_id": row["contest_id"], "handle": row["handle"], "code": "label_without_matching_review_event"})
    return {"run_id": run_id, "eligible_session_labels": counts, "review_events": events,
            "reviewed_sessions": reviewed_sessions, "issues": issues,
            "training_ready": False,
            "blocker": "Independent evidence assessment, adequate class support and frozen evaluation splits are required."}
