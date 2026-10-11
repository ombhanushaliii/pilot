"""Offline import of user-triggered source exports. No browser or network calls."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from .sources import (Archive, PROJECT, SourceError, exclusive_file, read_config,
                      source_url, utc_now, validate_loaded_snapshot)

MAX_EXPORT_BYTES = 10 * 1024 * 1024


def load_export(path, config):
    path = Path(path)
    with path.open("rb") as stream:
        body = stream.read(MAX_EXPORT_BYTES + 1)
    if len(body) > MAX_EXPORT_BYTES:
        raise SourceError("export_too_large")
    try:
        value = json.loads(body.decode("utf-8-sig"))
        if not isinstance(value, dict) or set(value) != {"version", "method", "source_url", "exported_at_utc", "snapshot"}:
            raise ValueError()
        if type(value["version"]) is not int or value["version"] != 1 or value["method"] != "user_click_loaded_dom":
            raise ValueError()
        if not isinstance(value["source_url"], str) or not isinstance(value["exported_at_utc"], str):
            raise ValueError()
        captured = datetime.fromisoformat(value["exported_at_utc"].replace("Z", "+00:00"))
        if captured.tzinfo is None or captured.utcoffset() != timedelta(0) or captured > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError()
        snapshot = value["snapshot"]
        required = {"title", "challenge", "source_count", "source", "rows"}
        if not isinstance(snapshot, dict) or not required <= set(snapshot) or set(snapshot) - required - {"source_text_mode"}:
            raise ValueError()
        if not isinstance(snapshot["title"], str) or type(snapshot["challenge"]) is not bool or type(snapshot["source_count"]) is not int:
            raise ValueError()
        if not isinstance(snapshot["source"], str) or not isinstance(snapshot["rows"], list):
            raise ValueError()
        for row in snapshot["rows"]:
            if not isinstance(row, dict) or set(row) != {"submission_href", "authors", "language", "problem_href"}:
                raise ValueError()
            if not all(isinstance(row[key], str) for key in ("submission_href", "language", "problem_href")):
                raise ValueError()
            if not isinstance(row["authors"], list) or not all(isinstance(x, str) for x in row["authors"]):
                raise ValueError()
    except (ValueError, TypeError, KeyError, OverflowError):
        raise SourceError("invalid_export_schema_or_timestamp") from None
    tasks = [task for task in config["submissions"] if source_url(task) == value["source_url"]]
    if len(tasks) != 1:
        raise SourceError("submission_not_in_manifest")
    task = tasks[0]
    source, identity = validate_loaded_snapshot(task, value["source_url"], snapshot)
    identity.update({"acquisition_method": value["method"], "exported_at_utc": value["exported_at_utc"],
                     "imported_at_utc": utc_now(), "export_file_sha256": hashlib.sha256(body).hexdigest(),
                     "http_status_observed": False})
    return task, source, identity


def import_export(path, config, archive):
    task, source, identity = load_export(path, config)
    with exclusive_file(archive.root / ".import-lock"):
        if archive.cached(task):
            old, metadata = archive.paths(task)
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if old.read_bytes() != source.encode("utf-8") or any(saved.get(key) != identity[key] for key in ("observed_handle", "problem_index", "language")):
                raise SourceError("export_conflicts_with_existing_source")
            status = "already_saved"
        else:
            archive.save(task, source, identity)
            status = "saved"
        archive.event({"submission_id": task["submission_id"], "status": status, "method": "user_click_loaded_dom"})
    return {"submission_id": task["submission_id"], "status": status, "label": "unknown", "network_requests": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "config" / "source-browser-export.json")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.manifest)
        # Reject invalid exports before creating any archive files.
        load_export(args.export, config)
        archive = Archive(PROJECT / "data" / "sources" / config["run_id"], config)
        print(json.dumps(import_export(args.export, config, archive), indent=2))
        return 0
    except (SourceError, ValueError, OSError) as error:
        print(f"Source import stopped: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
