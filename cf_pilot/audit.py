"""Machine-readable quality checks and an unlabelled review worksheet."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def audit_run(store, run_id, verify_raw=True):
    db = store.db
    run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run:
        raise ValueError("Unknown run_id")
    jobs = list(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)))
    sessions = list(db.execute("SELECT * FROM sessions WHERE run_id=?", (run_id,)))
    issues = []
    for job in jobs:
        if job["status"] != "complete":
            issues.append({"code": "incomplete_job", "job": job["job_key"], "status": job["status"]})
    if run["status"] not in {"complete"}:
        issues.append({"code": "run_not_complete", "status": run["status"], "error": run["error"]})
    if not sessions:
        issues.append({"code": "no_sessions"})
    config = json.loads(run["config_json"])
    handles = [row[0].casefold() for row in db.execute("SELECT handle FROM cohort WHERE run_id=? AND selected=1", (run_id,))]
    contest_ids = [row[0] for row in db.execute("SELECT contest_id FROM contests WHERE run_id=? AND selected=1", (run_id,))]
    contest_mode = config.get("sampling_strategy") == "contest"
    if contest_mode:
        from .contest_audit import audit_population
        panel = list(db.execute("SELECT contest_id FROM contest_panel WHERE run_id=?", (run_id,)))
        expected_jobs = {f"contest:{row[0]}" for row in panel}
        expected_sessions = db.execute("SELECT count(*) FROM cohort WHERE run_id=?", (run_id,)).fetchone()[0]
        expected_contests = 1
        population = audit_population(store, run_id, config, issues)
    else:
        expected_jobs = {f"history:{handle}" for handle in handles} | {
            f"session:{handle}:{identifier}" for handle in handles for identifier in contest_ids}
        expected_sessions = len(handles) * len(contest_ids)
        expected_contests = len(config["contest_ids"]) if config["contest_ids"] else config["contest_count"]
    missing_jobs = sorted(expected_jobs - {job["job_key"] for job in jobs})
    if missing_jobs:
        issues.append({"code": "missing_collection_jobs", "jobs": missing_jobs})
    if len(sessions) != expected_sessions:
        issues.append({"code": "missing_session_candidates", "expected": expected_sessions, "actual": len(sessions)})
    if len(contest_ids) != expected_contests:
        issues.append({"code": "contest_selection_incomplete", "expected": expected_contests, "actual": len(contest_ids)})
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        issues.append({"code": "sqlite_integrity"})
    conflicts = [dict(row) for row in db.execute("SELECT submission_id,detail_json FROM submission_conflicts WHERE run_id=?", (run_id,))]
    if conflicts:
        issues.append({"code": "changed_submission_snapshots", "count": len(conflicts), "examples": conflicts[:10]})
    if verify_raw:
        for row in db.execute("SELECT request_key FROM responses WHERE run_id=?", (run_id,)):
            try:
                store.cached(run_id, row[0])
            except (ValueError, OSError) as e:
                issues.append({"code": "raw_integrity", "detail": str(e)})
    selected = [dict(row) for row in db.execute(
        "SELECT contest_id,payload_json FROM contests WHERE run_id=? AND selected=1", (run_id,))]
    counts = {}
    for table in ["responses", "submissions", "rating_events", "standings", "sessions"]:
        counts[table] = db.execute(f"SELECT count(*) FROM {table} WHERE run_id=?", (run_id,)).fetchone()[0]
    counts["cohort_total"] = db.execute("SELECT count(*) FROM cohort WHERE run_id=?", (run_id,)).fetchone()[0]
    counts["cohort_selected"] = db.execute("SELECT count(*) FROM cohort WHERE run_id=? AND selected=1", (run_id,)).fetchone()[0]
    counts["eligible_sessions"] = sum(row["eligible"] for row in sessions)
    counts["eligible_sessions_with_prior_history"] = sum(bool(row["eligible"] and row["prior_contest_count"]) for row in sessions)
    counts["labels"] = dict(Counter(row[0] for row in db.execute("SELECT label FROM labels WHERE run_id=?", (run_id,))))
    missing = [dict(row) for row in sessions if row["eligible"] and not row["in_standings"]]
    # Model training is intentionally not implemented in this milestone.
    return {
        "run_id": run_id, "status": run["status"], "collection_checks_passed": not issues,
        "scope": "available target-contest submission census and fixed earlier-contest panel" if contest_mode else "targeted cohort pilot; not a complete contest participant census",
        **({"population": population} if contest_mode else {}),
        "counts": counts, "contests": [{"id": x["contest_id"], **json.loads(x["payload_json"])} for x in selected],
        "history_stop_reasons": dict(Counter(json.loads(job["detail"] or "{}").get("stop_reason")
                                             for job in jobs if job["job_key"].startswith("history:"))),
        "exclusions": dict(Counter(row["exclusion_reason"] for row in sessions if not row["eligible"])),
        "submission_sessions_absent_from_standings": missing, "issues": issues,
        "history_note": ("Coverage means participation in the fixed earlier-contest panel, not complete account history. All panel pages are exhausted. Earlier normalized submissions cover only the development sample; full raw pages are preserved."
                         if contest_mode else "Past verdicts are snapshots retrieved today. A lookback-boundary stop is documented, not proof that every historical record exists."),
        "training_ready": False, "training_blocker": "Evidence labels and frozen train/validation/test partitions are not established.",
    }


def write_reports(store, run_id, output: Path, report):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "data_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = list(store.db.execute("SELECT * FROM sessions WHERE run_id=? ORDER BY contest_id,handle", (run_id,)))
    with (output / "sessions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "contest_id", "handle", "eligible", "exclusion_reason", "submission_count",
                      "prior_contest_count", "history_complete", "in_standings"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    worksheet = output / "labels_for_review.csv"
    if not worksheet.exists() and report["status"] == "complete":
        with worksheet.open("w", newline="", encoding="utf-8") as f:
            columns = ["run_id", "contest_id", "handle", "label", "evidence_url", "evidence_type", "confidence", "reviewer_notes"]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(dict(row) for row in store.db.execute(
                "SELECT * FROM labels WHERE run_id=? ORDER BY contest_id,handle", (run_id,)))
    response_manifest = [dict(row) for row in store.db.execute(
        "SELECT request_key,url,params_json,raw_path,sha256,fetched_at,byte_count FROM responses WHERE run_id=? ORDER BY request_key", (run_id,))]
    manifest_bytes = json.dumps(response_manifest, sort_keys=True, indent=2).encode()
    (output / "response_manifest.json").write_bytes(manifest_bytes)
    (output / "response_manifest.sha256").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii")
    if "population" in report:
        coverage = [dict(row) for row in store.db.execute("SELECT * FROM participant_coverage WHERE run_id=? ORDER BY history_group,handle", (run_id,))]
        columns = ["run_id", "contest_id", "handle", "prior_contest_count", "history_group", "selected", "selection_probability"]
        for name, entries in [("participant_coverage.csv", coverage), ("development_sample.csv", [x for x in coverage if x["selected"]])]:
            with (output / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(entries)
        (output / "contest_panel.json").write_text(json.dumps(report["population"]["panel"], indent=2, ensure_ascii=False), encoding="utf-8")
