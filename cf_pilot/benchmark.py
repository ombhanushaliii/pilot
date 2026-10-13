"""Versioned, offline recorded-outcome benchmark; never writes source databases."""

from array import array
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median

from .features import quantile
from .outcomes import audit_outcomes
from .store import canonical


VERSION = "behaviour_only_v1"
FEATURE_GROUPS = {
    "basic": ["attempts", "attempted_problems", "first_submission_fraction",
              "last_submission_fraction", "attempts_per_hour"],
    "pacing": ["gap_min_fraction", "gap_p10_fraction", "gap_median_fraction",
               "gap_max_fraction", "gap_fraction_le_15s", "gap_fraction_le_60s",
               "max_attempts_in_60s"],
    "trajectory": ["retries", "retry_fraction", "retry_gap_min_fraction",
                   "retry_gap_median_fraction", "retry_fraction_le_60s",
                   "largest_problem_attempt_share", "problem_switch_fraction",
                   "problem_revisit_fraction"],
    "language": ["distinct_languages", "language_switch_fraction",
                 "within_problem_language_switch_fraction", "language_missing_fraction"],
}
FEATURE_COLUMNS = tuple(name for group in FEATURE_GROUPS.values() for name in group)
PARTITIONS = ("train", "validation", "test")
TABLE_COLUMNS = {
    "features": ("session_id",) + FEATURE_COLUMNS,
    "targets": ("session_id", "recorded_outcome"),
    "metadata": ("session_id", "handle", "contest_id", "participation_stratum"),
    "tracker": ("session_id", "handle", "contest_id", "split", "participation_stratum",
                "submission_count", "problem_count", "skipped_count", "final_attempts_skipped",
                "replacement_compatible_skips", "pattern", "recorded_outcome",
                "benchmark_exclusion", "misconduct_label"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def recorded_target(row):
    """Return proxy target or explicit exclusion, never misconduct truth."""
    if row["pattern"] == "outcome_unresolved":
        return None, "outcome_unresolved"
    if row["problem_count"] < 2:
        return None, "single_problem"
    if row["pattern"] == "all_skipped_multiple_problems":
        return 1, ""
    if row["pattern"] == "no_skipped_observed":
        return 0, ""
    return None, "ambiguous_skipped_pattern"


def behavior_features(items, contest):
    """Read only allowlisted transport fields. Judging and author fields are ignored."""
    start, duration = contest["startTimeSeconds"], contest["durationSeconds"]
    if type(duration) is not int or duration <= 0:
        raise ValueError("Invalid contest duration")
    sequence = sorted(items, key=lambda s: (s["creationTimeSeconds"], s["id"]))
    if not sequence or len({s["id"] for s in sequence}) != len(sequence):
        raise ValueError("Empty or duplicate session")
    grouped = defaultdict(list)
    for s in sequence:
        if s["contestId"] != contest["id"] or not start <= s["creationTimeSeconds"] < start + duration:
            raise ValueError("Feature session outside contest")
        grouped[s["problem"]["index"]].append(s)
    times = [s["creationTimeSeconds"] for s in sequence]
    gaps = [b - a for a, b in zip(times, times[1:])]
    retry_gaps = [b["creationTimeSeconds"] - a["creationTimeSeconds"]
                  for values in grouped.values() for a, b in zip(values, values[1:])]
    window, max_burst = deque(), 0
    for t in times:
        while window and t - window[0] > 60:
            window.popleft()
        window.append(t)
        max_burst = max(max_burst, len(window))
    seen, switches, revisits = set(), 0, 0
    previous = None
    for s in sequence:
        problem = s["problem"]["index"]
        if previous is not None and problem != previous:
            switches += 1
            revisits += problem in seen
        seen.add(problem)
        previous = problem
    languages = [s.get("programmingLanguage") for s in sequence]
    known = [x for x in languages if isinstance(x, str) and x.strip()]

    def language_fraction(values):
        pairs = [(a.get("programmingLanguage"), b.get("programmingLanguage"))
                 for a, b in values]
        valid = [(a, b) for a, b in pairs if isinstance(a, str) and a.strip()
                 and isinstance(b, str) and b.strip()]
        return sum(a != b for a, b in valid) / len(valid) if valid else None

    result = {
        "attempts": len(sequence), "attempted_problems": len(grouped),
        "first_submission_fraction": (times[0] - start) / duration,
        "last_submission_fraction": (times[-1] - start) / duration,
        "attempts_per_hour": len(sequence) * 3600 / duration,
        "gap_min_fraction": min(gaps) / duration if gaps else None,
        "gap_p10_fraction": quantile(gaps, .1) / duration if gaps else None,
        "gap_median_fraction": median(gaps) / duration if gaps else None,
        "gap_max_fraction": max(gaps) / duration if gaps else None,
        "gap_fraction_le_15s": sum(g <= 15 for g in gaps) / len(gaps) if gaps else None,
        "gap_fraction_le_60s": sum(g <= 60 for g in gaps) / len(gaps) if gaps else None,
        "max_attempts_in_60s": max_burst,
        "retries": len(retry_gaps), "retry_fraction": len(retry_gaps) / len(sequence),
        "retry_gap_min_fraction": min(retry_gaps) / duration if retry_gaps else None,
        "retry_gap_median_fraction": median(retry_gaps) / duration if retry_gaps else None,
        "retry_fraction_le_60s": sum(g <= 60 for g in retry_gaps) / len(retry_gaps) if retry_gaps else None,
        "largest_problem_attempt_share": max(map(len, grouped.values())) / len(sequence),
        "problem_switch_fraction": switches / len(gaps) if gaps else None,
        "problem_revisit_fraction": revisits / len(gaps) if gaps else None,
        "distinct_languages": len(set(known)),
        "language_switch_fraction": language_fraction(zip(sequence, sequence[1:])),
        "within_problem_language_switch_fraction": language_fraction(
            pair for values in grouped.values() for pair in zip(values, values[1:])),
        "language_missing_fraction": 1 - len(known) / len(sequence),
    }
    validate_features(result)
    return result


def validate_features(row):
    if set(row) != set(FEATURE_COLUMNS):
        raise ValueError("Feature allowlist mismatch")
    if any(value is not None and (type(value) not in {int, float} or not math.isfinite(value))
           for value in row.values()):
        raise ValueError("Non-finite or non-numeric feature")


def fit_preprocessor(rows):
    """Caller supplies training features only; targets and metadata are rejected."""
    values = {name: array("d") for name in FEATURE_COLUMNS}
    count = 0
    for row in rows:
        validate_features(row)
        count += 1
        for name, value in row.items():
            if value is not None:
                values[name].append(value)
    if not count:
        raise ValueError("No training rows")
    stats = {}
    for name, data in values.items():
        ordered = sorted(data)
        center = median(ordered) if ordered else 0.
        mean = math.fsum(ordered) / len(ordered) if ordered else 0.
        variance = math.fsum((x - mean) ** 2 for x in ordered) / len(ordered) if ordered else 0.
        stats[name] = {"observed": len(ordered), "missing": count - len(ordered),
                       "impute": center, "mean": mean, "scale": math.sqrt(variance) or 1.,
                       "quartiles": [quantile(ordered, q) if ordered else 0. for q in (.25, .5, .75)]}
    return {"feature_version": VERSION, "fitted_on": "train", "rows": count,
            "columns": list(FEATURE_COLUMNS), "stats": stats,
            "output_columns": list(FEATURE_COLUMNS) + [x + "__missing" for x in FEATURE_COLUMNS]}


def transform_features(row, preprocessor):
    validate_features(row)
    if preprocessor["columns"] != list(FEATURE_COLUMNS) or preprocessor["feature_version"] != VERSION:
        raise ValueError("Preprocessor schema mismatch")
    vector, mask = [], []
    for name in FEATURE_COLUMNS:
        stat, value = preprocessor["stats"][name], row[name]
        mask.append(int(value is None))
        vector.append(((stat["impute"] if value is None else value) - stat["mean"]) / stat["scale"])
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Non-finite transformed features")
    return vector + mask


def validate_partitions(config, contests):
    if set(config["splits"]) != set(PARTITIONS):
        raise ValueError("Missing partition")
    flattened = [cid for part in PARTITIONS for cid in config["splits"][part]]
    if any(not config["splits"][p] for p in PARTITIONS) or len(set(flattened)) != len(flattened):
        raise ValueError("Empty or overlapping contest splits")
    if set(flattened) != set(contests):
        raise ValueError("Partition does not cover source contest panel")
    for c in contests.values():
        audit_outcomes([], c, True, year=config["year"])
    for before, after in zip(PARTITIONS, PARTITIONS[1:]):
        end = max(contests[c]["startTimeSeconds"] + contests[c]["durationSeconds"]
                  for c in config["splits"][before])
        start = min(contests[c]["startTimeSeconds"] for c in config["splits"][after])
        if end >= start:
            raise ValueError("Partitions are not strictly chronological")


def source_manifest(db, config):
    run = config["source_run"]
    if not db.execute("SELECT 1 FROM runs WHERE run_id=? AND status='complete'", (run,)).fetchone():
        raise ValueError("Source run is not complete")
    contests = {cid: json.loads(payload) for cid, payload in db.execute(
        "SELECT c.contest_id,c.payload_json FROM contests c JOIN contest_panel p USING(run_id,contest_id) WHERE c.run_id=?", (run,))}
    validate_partitions(config, contests)
    pages = defaultdict(list)
    for key, url, params, path, sha, size in db.execute(
            "SELECT request_key,url,params_json,raw_path,sha256,byte_count FROM responses WHERE run_id=?", (run,)):
        params = json.loads(params)
        if url == "https://codeforces.com/api/contest.status" and params.get("contestId") in contests:
            if type(params.get("from")) is not int:
                raise ValueError("Invalid page offset")
            pages[params["contestId"]].append({"key": key, "params": params, "raw_path": path,
                                              "sha256": sha, "bytes": size})
    result = {}
    for cid, contest in contests.items():
        job = db.execute("SELECT status,pages,detail FROM jobs WHERE run_id=? AND job_key=?",
                         (run, f"contest:{cid}")).fetchone()
        detail = json.loads(job[2] or "{}") if job else {}
        if not job or job[0] != "complete" or detail.get("stop_reason") != "exhausted":
            raise ValueError(f"Incomplete source contest {cid}")
        archive_pages = sorted(pages[cid], key=lambda r: r["params"]["from"])
        if len(archive_pages) != job[1] or not archive_pages or archive_pages[0]["params"]["from"] != 1:
            raise ValueError("Source page manifest mismatch")
        result[str(cid)] = {"contest": contest, "job_detail": detail, "pages": archive_pages}
    return result


def replay_archives(root, source, counter):
    """Verify every raw response and replay only the frozen snapshot ID range."""
    root = Path(root).resolve()
    highest = source["job_detail"].get("highest_id")
    if highest is not None and type(highest) is not int:
        raise ValueError("Invalid snapshot highest ID")
    for page in source["pages"]:
        path = (root / page["raw_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Archive path outside source root")
        body = gzip.decompress(path.read_bytes())
        if len(body) != page["bytes"] or hashlib.sha256(body).hexdigest() != page["sha256"]:
            raise ValueError("Archived response checksum mismatch")
        payload = json.loads(body)
        if payload.get("status") != "OK" or not isinstance(payload.get("result"), list):
            raise ValueError("Invalid archived API response")
        items = payload["result"]
        ids = [s.get("id") for s in items]
        if any(type(sid) is not int for sid in ids) or any(a <= b for a, b in zip(ids, ids[1:])):
            raise ValueError("Unordered or invalid archive IDs")
        if page["params"]["from"] == 1:
            # Earlier collector checkpoints predate the explicit highest-ID field.
            # Recover it from the verified, immutable first page, never later pages.
            if not ids:
                raise ValueError("Source contest has no first-page submissions")
            if highest is not None and highest != ids[0]:
                raise ValueError("Snapshot highest ID differs from first-page archive")
            highest = ids[0]
        if highest is None:
            raise ValueError("Missing verified first-page snapshot boundary")
        counter["responses_verified"] += 1
        if not items:
            counter["empty_pages"] += 1
        for item in items:
            if item["id"] > highest:
                counter["above_snapshot_id"] += 1
                continue
            yield item
    if not counter["empty_pages"]:
        raise ValueError("No archived empty end page")


def csv_writer(path, columns):
    stream = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    return stream, writer


def build_contest(root, destination, source, cid, split, year):
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "complete.json"
    digest = hashlib.sha256(canonical(source).encode()).hexdigest()
    if marker.exists():
        result = json.loads(marker.read_text(encoding="utf-8"))
        if result["source_digest"] != digest or result["feature_version"] != VERSION or result["year"] != year or result["split"] != split:
            raise ValueError("Completed contest differs from frozen source")
        for name, sha in result["artifacts"].items():
            if file_hash(destination / name) != sha:
                raise ValueError("Completed benchmark artifact checksum mismatch")
        return result
    feature_map, replay = {}, Counter()
    contest = source["contest"]

    def receive(handle, items):
        if len({s["problem"]["index"] for s in items}) >= 2:
            feature_map[handle] = behavior_features(items, contest)

    summary, sessions = audit_outcomes(replay_archives(root, source, replay), contest, True,
                                       year=year, on_session=receive)
    if summary["raw_unique_submissions"] != source["job_detail"]["unique_items"]:
        raise ValueError("Replayed raw census differs from source checkpoint")
    streams, writers, counts, strata = [], {}, Counter(), defaultdict(Counter)
    try:
        for table, columns in TABLE_COLUMNS.items():
            stream, writer = csv_writer(destination / (table + ".csv.tmp"), columns)
            streams.append(stream)
            writers[table] = writer
        for row in sessions:
            target, exclusion = recorded_target(row)
            key = f"{cid}:{row['handle']}"
            tracker = {name: row[name] for name in TABLE_COLUMNS["tracker"] if name in row}
            tracker.update(session_id=key, split=split, recorded_outcome=target, benchmark_exclusion=exclusion)
            writers["tracker"].writerow(tracker)
            if target is None:
                counts["excluded:" + exclusion] += 1
                continue
            values = feature_map[row["handle"]]
            writers["features"].writerow({"session_id": key} | values)
            writers["targets"].writerow({"session_id": key, "recorded_outcome": target})
            writers["metadata"].writerow({"session_id": key, **{n: row[n] for n in ("handle", "contest_id", "participation_stratum")}})
            label = "positive" if target else "comparison"
            counts[label] += 1
            strata[row["participation_stratum"]][label] += 1
    finally:
        for stream in streams:
            stream.close()
    artifacts = {}
    for table in TABLE_COLUMNS:
        path = destination / (table + ".csv")
        path.with_suffix(path.suffix + ".tmp").replace(path)
        artifacts[path.name] = file_hash(path)
    result = {"contest_id": cid, "split": split, "year": year, "feature_version": VERSION,
              "source_digest": digest, "audit": summary, "replay": dict(replay),
              "benchmark_counts": dict(counts), "participation_counts": {k: dict(v) for k, v in strata.items()},
              "artifacts": artifacts}
    write_json(marker, result)
    return result


def merge_partitions(destination, config):
    result, seen_all, handle_sets = {}, set(), {}
    for split in PARTITIONS:
        target = destination / split
        target.mkdir(exist_ok=True)
        streams, writers, counts, handles, strata, seen = [], {}, Counter(), set(), defaultdict(Counter), set()
        try:
            for table in ("features", "targets", "metadata"):
                stream, writer = csv_writer(target / (table + ".csv.tmp"), TABLE_COLUMNS[table])
                streams.append(stream)
                writers[table] = writer
            for cid in config["splits"][split]:
                folder = destination / "contests" / str(cid)
                readers = []
                opened = []
                try:
                    for table in ("features", "targets", "metadata"):
                        s = (folder / (table + ".csv")).open(encoding="utf-8", newline="")
                        opened.append(s)
                        reader = csv.DictReader(s)
                        if reader.fieldnames != list(TABLE_COLUMNS[table]):
                            raise ValueError("Table column mismatch")
                        readers.append(reader)
                    from itertools import zip_longest
                    for feature, label, meta in zip_longest(*readers):
                        if None in (feature, label, meta) or len({x["session_id"] for x in (feature, label, meta)}) != 1:
                            raise ValueError("Feature/target/metadata join mismatch")
                        key = feature["session_id"]
                        if key in seen or key in seen_all:
                            raise ValueError("Session overlaps partitions")
                        seen.add(key)
                        if key != f"{cid}:{meta['handle']}" or int(meta["contest_id"]) != cid:
                            raise ValueError("Session identity mismatch")
                        value = int(label["recorded_outcome"])
                        if value not in (0, 1):
                            raise ValueError("Invalid target")
                        counts["positive" if value else "comparison"] += 1
                        strata[meta["participation_stratum"]][str(value)] += 1
                        handles.add(meta["handle"])
                        for table, row in zip(("features", "targets", "metadata"), (feature, label, meta)):
                            writers[table].writerow(row)
                finally:
                    for s in opened:
                        s.close()
        finally:
            for s in streams:
                s.close()
        if not counts["positive"] or not counts["comparison"]:
            raise ValueError("Partition lacks an outcome class")
        seen_all.update(seen)
        artifacts = {}
        for table in writers:
            path = target / (table + ".csv")
            path.with_suffix(path.suffix + ".tmp").replace(path)
            artifacts[path.name] = file_hash(path)
        result[split] = {"rows": len(seen), "class_counts": dict(counts), "unique_handles": len(handles),
                         "participation_counts": {k: dict(v) for k, v in strata.items()}, "artifacts": artifacts}
        handle_sets[split] = handles
    result["user_overlap"] = {f"{a}:{b}": len(handle_sets[a] & handle_sets[b])
                              for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    train_seen = handle_sets["train"]
    ever_seen = train_seen | handle_sets["validation"]
    unseen = Counter()
    with (destination / "test" / "metadata.csv").open(encoding="utf-8") as m, (destination / "test" / "targets.csv").open(encoding="utf-8") as t:
        for meta, label in zip(csv.DictReader(m), csv.DictReader(t)):
            if meta["handle"] not in train_seen:
                unseen["unseen_in_train:" + label["recorded_outcome"]] += 1
            if meta["handle"] not in ever_seen:
                unseen["unseen_in_train_and_validation:" + label["recorded_outcome"]] += 1
    result["test_unseen_user_counts"] = dict(unseen)
    return result


def numeric_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(TABLE_COLUMNS["features"]):
            raise ValueError("Training feature schema mismatch")
        for row in reader:
            yield {name: float(row[name]) if row[name] != "" else None for name in FEATURE_COLUMNS}


def build_benchmark(project_root, config, progress=print):
    root = Path(project_root).resolve()
    source_root = (root / config["source_data"]).resolve()
    source_db = source_root / "pilot.sqlite"
    destination = root / "data" / config["benchmark_id"]
    destination.mkdir(parents=True, exist_ok=True)
    frozen = destination / "manifest.json"
    progress("Checking source database fingerprint", flush=True)
    source_sha = file_hash(source_db)
    db = sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)
    try:
        sources = source_manifest(db, config)
    finally:
        db.close()
    manifest = {"config": config, "source_database_sha256": source_sha,
                "feature_version": VERSION, "feature_groups": FEATURE_GROUPS,
                "source_contests": sources, "target_name": "all_attempts_skipped_multiple_problems",
                "target_is_misconduct_truth": False, "network_requests": 0}
    if frozen.exists():
        if canonical(json.loads(frozen.read_text(encoding="utf-8"))) != canonical(manifest):
            raise ValueError("Frozen benchmark differs; use a new benchmark version")
    else:
        write_json(frozen, manifest)
    status = destination / "build-status.json"
    write_json(status, {"status": "building"})
    try:
        summaries = []
        for split in PARTITIONS:
            for cid in config["splits"][split]:
                progress(f"Auditing {split} contest {cid}", flush=True)
                s = build_contest(source_root, destination / "contests" / str(cid), sources[str(cid)], cid, split, config["year"])
                summaries.append(s)
                progress(f"Contest {cid}: {s['benchmark_counts']}", flush=True)
        partitions = merge_partitions(destination, config)
        train_file = destination / "train" / "features.csv"
        preprocessing = fit_preprocessor(numeric_rows(train_file))
        preprocessing["training_features_sha256"] = file_hash(train_file)
        write_json(destination / "preprocessor.json", preprocessing)
        for split in PARTITIONS:
            for row in numeric_rows(destination / split / "features.csv"):
                transform_features(row, preprocessing)
        if file_hash(source_db) != source_sha:
            raise ValueError("Source database changed during benchmark build")
        summary = {"benchmark_id": config["benchmark_id"], "complete": True,
                   "feature_version": VERSION, "feature_columns": list(FEATURE_COLUMNS),
                   "partitions": partitions, "contests": summaries,
                   "source_database_unchanged": True, "network_requests": 0,
                   "misconduct_labels_assigned": 0, "preprocessing_fitted_on": "train",
                   "preprocessor_sha256": file_hash(destination / "preprocessor.json")}
        write_json(destination / "summary.json", summary)
        report = root / "reports" / config["benchmark_id"]
        report.mkdir(parents=True, exist_ok=True)
        write_json(report / "summary.json", summary)
        write_json(status, {"status": "complete"})
        return summary
    except BaseException as error:
        write_json(status, {"status": "failed", "error": str(error)})
        raise
