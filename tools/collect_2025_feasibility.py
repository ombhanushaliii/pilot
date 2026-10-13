"""Frozen six-contest audit. Run from D:/pilot; no models or labels are changed."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cf_pilot.api import Client, CollectionError
from cf_pilot.collect import collect_pages
from cf_pilot.outcomes import audit_outcomes
from cf_pilot.store import Store, canonical

ROOT = Path(__file__).resolve().parents[1]
RUN = "api-2025-feasibility-001"
REPORT = ROOT / "reports" / RUN
REPORT.mkdir(parents=True, exist_ok=True)
MANIFEST = REPORT / "manifest.json"

config = json.loads((ROOT / "config/api-2025-feasibility.json").read_text(encoding="utf-8"))
if not MANIFEST.exists():
    MANIFEST.write_text(json.dumps(config, indent=2), encoding="utf-8")
elif canonical(json.loads(MANIFEST.read_text(encoding="utf-8"))) != canonical(config):
    raise ValueError("Frozen manifest differs from configuration; use a separate run")
for choice in config["contests"]:
    audit_outcomes([], choice["metadata"], False)

store = Store(ROOT / "data" / RUN)
store.start_run(RUN, config)
client = Client(store, RUN, interval=config["request_interval_seconds"],
                timeout=config["timeout_seconds"], max_attempts=config["max_attempts"])
summaries = []

def sink(items, key):
    for item in items:
        existing = store.db.execute("SELECT payload_json FROM submissions WHERE run_id=? AND submission_id=?",
                                    (RUN, item["id"])).fetchone()
        if existing and canonical(json.loads(existing[0])) != canonical(item):
            raise CollectionError("Conflicting overlapping submission; start a new snapshot instead of merging")
    store.ingest_submissions(RUN, key, items)

try:
    for choice in config["contests"]:
        c = choice["metadata"]
        store.db.execute("INSERT OR IGNORE INTO contests VALUES(?,?,?,1)", (RUN, c["id"], canonical(c)))
        store.db.commit()
        complete = collect_pages(client, f"contest:{c['id']}", "contest.status", {"contestId": c["id"]},
                                 config["page_size"], config["max_pages_per_contest"],
                                 item_sink=sink, exhaust_on_empty=True,
                                 progress=lambda msg: print(msg, flush=True))
        job = store.job(RUN, f"contest:{c['id']}")
        records = (json.loads(row[0]) for row in store.db.execute(
            "SELECT payload_json FROM submissions WHERE run_id=? AND contest_id=?", (RUN, c["id"])))
        summary, sessions = audit_outcomes(records, c, complete)
        expected = json.loads(job["detail"])["unique_items"]
        if expected != summary["raw_unique_submissions"]:
            raise CollectionError("Raw census and normalized row counts disagree")
        summary["pages"] = job["pages"]
        summary["snapshot_highest_id"] = json.loads(job["detail"])["highest_id"]
        summary["contest_inclusion_probability"] = choice["contest_inclusion_probability"]
        (REPORT / f"contest-{c['id']}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (REPORT / f"contest-{c['id']}-sessions.json").write_text(json.dumps(sessions), encoding="utf-8")
        summaries.append(summary)
        (REPORT / "summary.json").write_text(json.dumps({"run_id": RUN,
            "complete": len(summaries) == 6 and all(s["collection_complete"] for s in summaries),
            "completed_contests": summaries, "network_requests_this_invocation": client.network_requests,
            "misconduct_labels_assigned": 0}, indent=2), encoding="utf-8")
        print("OUTCOME AUDIT " + json.dumps(summary), flush=True)
        if not complete:
            raise CollectionError("Contest pagination capped; do not treat this as a complete session census")
    store.db.execute("UPDATE runs SET status='complete',error=NULL WHERE run_id=?", (RUN,))
    store.db.commit()
except BaseException as error:
    store.db.execute("UPDATE runs SET status='failed',error=? WHERE run_id=?", (str(error), RUN))
    store.db.commit()
    raise
finally:
    store.close()
