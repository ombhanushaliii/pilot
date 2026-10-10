import argparse
import json
import sqlite3
from pathlib import Path
import sys

from .api import Client
from .audit import audit_run, write_reports
from .collect import collect, load_config
from .sessions import build_sessions
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-contest Codeforces pilot collector")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("collect", help="Collect or resume a frozen pilot run")
    download.add_argument("--config", type=Path, default=Path("config/pilot.json"))
    check = commands.add_parser("check", help="Verify an existing dataset and refresh reports")
    check.add_argument("--run-id", default="pilot-001")
    commands.add_parser("status", help="List run state and incomplete jobs")
    args = parser.parse_args(argv)
    store = Store(args.data_dir)
    try:
        if args.command == "collect":
            config = load_config(args.config)
            run_id = config["run_id"]
            store.start_run(run_id, config)
            client = Client(store, run_id, config["request_interval_seconds"],
                            config["request_timeout_seconds"], config["max_attempts"])
            error = None
            try:
                complete = collect(client, config, progress=lambda text: print(text, flush=True))
                build_sessions(store, run_id, config["history_lookback_days"])
                status = "complete" if complete else "incomplete"
            except (Exception, KeyboardInterrupt) as e:
                error = str(e) or "Interrupted by user"
                status = "failed" if not isinstance(e, KeyboardInterrupt) else "interrupted"
                # Build whatever was successfully collected, but mark the run incomplete.
                build_sessions(store, run_id, config["history_lookback_days"])
            store.db.execute("UPDATE runs SET status=?,error=? WHERE run_id=?", (status, error, run_id))
            store.db.commit()
            report = audit_run(store, run_id)
            write_reports(store, run_id, args.reports_dir / run_id, report)
            print(json.dumps({"status": status, "network_requests": client.network_requests,
                              "cache_hits": client.cache_hits, "counts": report["counts"],
                              "error": error, "reports": str(args.reports_dir / run_id)}, indent=2))
            return 0 if report["collection_checks_passed"] else 2
        if args.command == "check":
            report = audit_run(store, args.run_id)
            write_reports(store, args.run_id, args.reports_dir / args.run_id, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["collection_checks_passed"] else 2
        runs = [dict(row) for row in store.db.execute("SELECT run_id,status,created_at,error FROM runs ORDER BY created_at")]
        pending = [dict(row) for row in store.db.execute("SELECT run_id,job_key,status,pages,next_from FROM jobs WHERE status!='complete'")]
        print(json.dumps({"runs": runs, "incomplete_jobs": pending}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        store.close()
