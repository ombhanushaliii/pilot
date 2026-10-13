"""Offline, checksummed history construction from every eligible raw session."""
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import behavior_features, file_hash, replay_archives, write_json, FEATURE_COLUMNS
from cf_pilot.context import compact_history, history_features, CONTEXT_COLUMNS
from cf_pilot.outcomes import audit_outcomes
from cf_pilot.store import canonical


def frozen(path, value):
    if path.exists() and canonical(json.loads(path.read_text(encoding="utf-8"))) != canonical(value):
        raise ValueError("Frozen experiment differs: " + str(path))
    if not path.exists():
        write_json(path, value)


def prepare():
    original = ROOT / "data/deep-benchmark-2026-001"
    source = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    base = ROOT / "data/recommendation-tests-001"
    base.mkdir(exist_ok=True)
    roots = {"contest-002": ROOT / "data",
             "api-2025-feasibility-001": ROOT / "data/api-2025-feasibility-001",
             "rl-expansion-2026-002": ROOT / "data/rl-expansion-2026-002"}
    for name, path in roots.items():
        if file_hash(path / "pilot.sqlite") != source["source_fingerprints"][name]:
            raise ValueError("Source DB changed")
    ordered = sorted(source["source_contests"],
                     key=lambda cid: source["source_contests"][cid]["contest"]["startTimeSeconds"])
    old = json.loads((ROOT / "config/benchmark-2026.json").read_text())
    ids2025 = [int(cid) for cid in ordered if datetime.fromtimestamp(
        source["source_contests"][cid]["contest"]["startTimeSeconds"], timezone.utc).year == 2025]
    config = {"run_id": "recommendation-tests-001", "source_benchmark": "deep-benchmark-2026-001",
        "network_requests": 0, "history_last_contests": 20,
        "history_population": "All in-window individual CONTESTANT/OUT_OF_COMPETITION sessions, regardless of outcome or problem count",
        "history_input_columns": list(CONTEXT_COLUMNS),
        "folds": {
            "earlier": {"train": ids2025 + old["splits"]["train"], "validation": old["splits"]["validation"]},
            "later": {"train": source["config"]["splits"]["train"], "validation": source["config"]["splits"]["validation"]}},
        "diagnostic_contests": [2274], "diagnostic_is_untouched": False,
        "candidates": [
            {"name": "baseline", "history": False, "full_visible": False, "cost": .005},
            {"name": "free_inspection", "history": False, "full_visible": False, "cost": 0.},
            {"name": "all_visible", "history": False, "full_visible": True, "cost": 0.},
            {"name": "history", "history": True, "full_visible": False, "cost": .005},
            {"name": "history_all_visible", "history": True, "full_visible": True, "cost": 0.}],
        "settings": {"updates": 18000, "validation_interval": 2000, "environments": 64,
                     "batch_size": 256, "replay_per_class": 50000, "warmup_updates": 500,
                     "learning_rate": .001, "target_tau": .01, "false_positive_penalty": .5,
                     "width": 64, "search_seed": 42, "repeat_seeds": [736, 2026]},
        "checkpoint_selection": "Mean per-contest validation F1, then worst-contest F1, then pooled F1",
        "pooled_checkpoint_control": "Also preserve best pooled-F1 weights from the identical learning run",
        "candidate_selection": "Mean macro-contest F1 over both development folds for seed42; repeat winner and baseline; compare repeat means",
        "representative_policy_seed": 42,
        "initial_history_visibility": "Cached history is initially visible for history candidates; baseline uses zero history placeholders to match parameter count",
        "all_visible_reference": "One-step contextual-bandit Q learner, with all groups initially provided and only two terminal actions",
        "budgets": "Equal updates and transitions; episode counts differ with inspection policy",
        "active_model_changed": False}
    frozen(ROOT / "config/recommendation-tests.json", config)
    frozen(base / "manifest.json", {"config": config, "source_manifest_sha256": file_hash(original / "manifest.json"),
           "source_databases": source["source_fingerprints"],
           "code_sha256": {name: file_hash(ROOT / name) for name in ["cf_pilot/context.py", "tools/prepare_recommendation_tests.py"]}})
    for left in config["folds"].values():
        train, val = left["train"], left["validation"]
        end = max(source["source_contests"][str(c)]["contest"]["startTimeSeconds"] +
                  source["source_contests"][str(c)]["contest"]["durationSeconds"] for c in train)
        start = min(source["source_contests"][str(c)]["contest"]["startTimeSeconds"] for c in val)
        if set(train) & set(val) or end >= start:
            raise ValueError("Fold overlap or chronology error")
    source_roots = {}
    for cid in ordered:
        year = datetime.fromtimestamp(source["source_contests"][cid]["contest"]["startTimeSeconds"], timezone.utc).year
        source_roots[cid] = roots["api-2025-feasibility-001" if year == 2025 else
                                 "rl-expansion-2026-002" if cid == "2274" else "contest-002"]
    # Cache raw-session features before looking at any binary benchmark targets.
    for cid in ordered:
        folder = base / "raw-behavior" / cid
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "complete.json"
        if marker.exists():
            if file_hash(folder / "features.csv") != json.loads(marker.read_text())["sha256"]:
                raise ValueError("Raw feature cache changed")
            continue
        record = source["source_contests"][cid]
        contest = record["contest"]
        counter, count = Counter(), [0]
        with (folder / "features.csv.tmp").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["handle", *FEATURE_COLUMNS], lineterminator="\n")
            writer.writeheader()
            def receive(handle, items):
                writer.writerow({"handle": handle, **behavior_features(items, contest)})
                count[0] += 1
            audit, _ = audit_outcomes(replay_archives(source_roots[cid], record, counter), contest,
                True, year=datetime.fromtimestamp(contest["startTimeSeconds"], timezone.utc).year, on_session=receive)
        if audit["raw_unique_submissions"] != record["job_detail"]["unique_items"] or audit["sessions"] != count[0]:
            raise ValueError("Raw history census differs")
        (folder / "features.csv.tmp").replace(folder / "features.csv")
        write_json(marker, {"rows": count[0], "sha256": file_hash(folder / "features.csv"), "replay": dict(counter)})
        print(f"Raw verdict-free history features: contest {cid}, {count[0]:,} sessions", flush=True)
    history, counts = defaultdict(lambda: deque(maxlen=20)), {}
    for cid in ordered:
        contest = source["source_contests"][cid]["contest"]
        start, end = contest["startTimeSeconds"], contest["startTimeSeconds"] + contest["durationSeconds"]
        old_folder = original / "contests" / cid
        old_marker = json.loads((old_folder / "complete.json").read_text())
        for name in ("features.csv", "targets.csv", "metadata.csv"):
            if file_hash(old_folder / name) != old_marker["artifacts"][name]:
                raise ValueError("Original primary benchmark changed")
        with (old_folder / "features.csv").open(encoding="utf-8", newline="") as stream:
            primary = {row["session_id"]: row for row in csv.DictReader(stream)}
        folder = base / "contests" / cid
        folder.mkdir(parents=True, exist_ok=True)
        n, seen = 0, set()
        with ((folder / "features.csv.tmp").open("w", encoding="utf-8", newline="") as output,
              (base / "raw-behavior" / cid / "features.csv").open(encoding="utf-8", newline="") as raw):
            writer = csv.DictWriter(output, fieldnames=["session_id", *CONTEXT_COLUMNS], lineterminator="\n")
            writer.writeheader()
            for row in csv.DictReader(raw):
                handle = row.pop("handle")
                if handle in seen:
                    raise ValueError("Duplicate current-contest handle")
                seen.add(handle)
                values = {name: float(row[name]) if row[name] else None for name in FEATURE_COLUMNS}
                past = history[handle]
                extra = history_features(values, past, start)
                key = f"{cid}:{handle}"
                if key in primary:
                    old_values = {name: float(primary[key][name]) if primary[key][name] else None for name in FEATURE_COLUMNS}
                    if old_values != values:
                        raise ValueError("Current behavioural values differ from original")
                    writer.writerow({"session_id": key, **values, **extra})
                    n += 1
                past.append(compact_history(values, end))
        if n != len(primary):
            raise ValueError("Primary session join incomplete")
        path = folder / "features.csv"
        temp = folder / "features.csv.tmp"
        if path.exists() and file_hash(path) != file_hash(temp):
            raise ValueError("Previously built context features changed")
        temp.replace(path)
        counts[cid] = {"rows": n, "sha256": file_hash(path),
                       "target_sha256": file_hash(old_folder / "targets.csv"),
                       "metadata_sha256": file_hash(old_folder / "metadata.csv")}
        print(f"Strictly earlier context: contest {cid}, {n:,} primary sessions", flush=True)
    for name, path in roots.items():
        if file_hash(path / "pilot.sqlite") != source["source_fingerprints"][name]:
            raise ValueError("Source DB changed during offline preparation")
    frozen(base / "complete.json", {"complete": True, "network_requests": 0,
           "source_databases_unchanged": True, "contests": counts})
    print("Offline contextual benchmark complete; no network requests", flush=True)


if __name__ == "__main__":
    prepare()
