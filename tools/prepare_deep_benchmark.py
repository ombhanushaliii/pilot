"""Combine archived years and reserve a newly collected contest for final testing."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import (FEATURE_COLUMNS, FEATURE_GROUPS, VERSION, build_contest,
    file_hash, fit_preprocessor, merge_partitions, numeric_rows, transform_features, write_json)
from cf_pilot.store import canonical


def source(root, run, ids):
    db = sqlite3.connect((root / "pilot.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT status FROM runs WHERE run_id=?", (run,)).fetchone()
        if not row or row[0] != "complete":
            raise ValueError("Source collection incomplete")
        records = {}
        for cid in ids:
            metadata = json.loads(db.execute("SELECT payload_json FROM contests WHERE run_id=? AND contest_id=?", (run, cid)).fetchone()[0])
            job = db.execute("SELECT status,pages,detail FROM jobs WHERE run_id=? AND job_key=?", (run, f"contest:{cid}")).fetchone()
            detail = json.loads(job[2])
            if job[0] != "complete" or detail["stop_reason"] != "exhausted":
                raise ValueError("Source contest not exhausted")
            pages = []
            for key, params, path, sha, size in db.execute("SELECT request_key,params_json,raw_path,sha256,byte_count FROM responses WHERE run_id=? AND url='https://codeforces.com/api/contest.status'", (run,)):
                params = json.loads(params)
                if params.get("contestId") == cid:
                    pages.append({"key": key, "params": params, "raw_path": path, "sha256": sha, "bytes": size})
            pages.sort(key=lambda p: p["params"]["from"])
            if len(pages) != job[1]:
                raise ValueError("Source page manifest incomplete")
            records[cid] = {"contest": metadata, "job_detail": detail, "pages": pages}
        return records
    finally:
        db.close()


def prepare():
    previous = json.loads((ROOT / "config/benchmark-2026.json").read_text(encoding="utf-8"))
    old_ids = previous["splits"]["train"] + previous["splits"]["validation"] + previous["splits"]["test"]
    ids2025 = [2063, 2078, 2107, 2119, 2160, 2163]
    specs = [(ROOT / "data", "contest-002", old_ids),
             (ROOT / "data/api-2025-feasibility-001", "api-2025-feasibility-001", ids2025),
             (ROOT / "data/rl-expansion-2026-002", "rl-expansion-2026-002", [2274])]
    sources, locations, fingerprints = {}, {}, {}
    for root, run, ids in specs:
        print("Verifying source " + run, flush=True)
        fingerprints[run] = file_hash(root / "pilot.sqlite")
        for cid, record in source(root, run, ids).items():
            if cid in sources:
                raise ValueError("Contest duplicated across sources")
            sources[cid], locations[cid] = record, root
    train = sorted(ids2025 + previous["splits"]["train"] + previous["splits"]["validation"],
                   key=lambda cid: sources[cid]["contest"]["startTimeSeconds"])
    config = {"benchmark_id": "deep-benchmark-2026-001", "splits": {
        "train": train, "validation": previous["splits"]["test"], "test": [2274]},
        "policy": "all_skipped_multiple_problems_vs_no_skipped_v1",
        "former_test_reclassified_as": "validation development data; no longer an untouched test",
        "fresh_test_stability": "finished recent snapshot, administrative verdicts may change"}
    for before, after in (("train", "validation"), ("validation", "test")):
        end = max(sources[c]["contest"]["startTimeSeconds"] + sources[c]["contest"]["durationSeconds"] for c in config["splits"][before])
        start = min(sources[c]["contest"]["startTimeSeconds"] for c in config["splits"][after])
        if end >= start:
            raise ValueError("Not a strictly temporal partition")
    destination = ROOT / "data" / config["benchmark_id"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "source_fingerprints": fingerprints, "source_contests": sources,
                "feature_version": VERSION, "feature_groups": FEATURE_GROUPS}
    frozen = destination / "manifest.json"
    if frozen.exists() and canonical(json.loads(frozen.read_text(encoding="utf-8"))) != canonical(manifest):
        raise ValueError("Expanded frozen benchmark differs")
    write_json(frozen, manifest)
    summaries = []
    old_folder = ROOT / "data/benchmark-2026-001/contests"
    for split, ids in config["splits"].items():
        for cid in ids:
            target = destination / "contests" / str(cid)
            old = old_folder / str(cid)
            if old.exists() and not (target / "complete.json").exists():
                saved = json.loads((old / "complete.json").read_text(encoding="utf-8"))
                if saved["source_digest"] != hashlib.sha256(canonical(sources[cid]).encode()).hexdigest():
                    raise ValueError("Cached source fingerprint differs")
                target.mkdir(parents=True, exist_ok=True)
                for name, sha in saved["artifacts"].items():
                    if file_hash(old / name) != sha:
                        raise ValueError("Original benchmark artifact changed")
                    shutil.copyfile(old / name, target / name)
                if saved["split"] != split:
                    with (target / "tracker.csv").open(encoding="utf-8", newline="") as stream:
                        reader = csv.DictReader(stream)
                        columns, rows = reader.fieldnames, list(reader)
                    with (target / "tracker.csv").open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
                        writer.writeheader()
                        for row in rows:
                            row["split"] = split
                            writer.writerow(row)
                saved["split"] = split
                saved["artifacts"] = {name: file_hash(target / name) for name in saved["artifacts"]}
                saved["derived_from"] = str(old.relative_to(ROOT))
                write_json(target / "complete.json", saved)
            year = 2025 if cid in ids2025 else 2026
            print(f"Preparing {split} contest {cid}", flush=True)
            summaries.append(build_contest(locations[cid], target, sources[cid], cid, split, year))
    partitions = merge_partitions(destination, config)
    pre = fit_preprocessor(numeric_rows(destination / "train/features.csv"))
    pre["training_features_sha256"] = file_hash(destination / "train/features.csv")
    write_json(destination / "preprocessor.json", pre)
    for split in ("train", "validation", "test"):
        for row in numeric_rows(destination / split / "features.csv"):
            transform_features(row, pre)
    for root, run, ids in specs:
        if file_hash(root / "pilot.sqlite") != fingerprints[run]:
            raise ValueError("Source changed during expanded build")
    result = {"benchmark_id": config["benchmark_id"], "complete": True, "feature_version": VERSION,
              "feature_columns": list(FEATURE_COLUMNS), "partitions": partitions, "contests": summaries,
              "preprocessor_sha256": file_hash(destination / "preprocessor.json"), "source_databases_unchanged": True,
              "misconduct_labels_assigned": 0, "fresh_test": [2274], "previous_test_is_now_validation": True}
    write_json(destination / "summary.json", result)
    write_json(ROOT / "config/deep-benchmark-2026.json", config)
    print(json.dumps({"complete": True, "partitions": partitions}, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
