"""Deterministic post-contest timing features, without labels or standings."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .audit import audit_run
from .sessions import eligibility
from .store import canonical


VERSION = "timing_panel_v1"
MIN_HISTORY = 5
BASE_COLUMNS = (
    "attempts", "attempted_problems", "retries", "retry_gap_min_seconds",
    "retry_gap_p10_seconds", "retry_gap_median_seconds", "retry_fraction_le_15s",
    "retry_fraction_le_60s", "first_submission_fraction", "last_submission_fraction",
    "attempts_per_hour", "language_switches", "problem_rating_mean",
    "problem_rating_max", "problem_rating_missing_fraction",
)
HISTORY_COLUMNS = tuple("prior_" + name + suffix for name in BASE_COLUMNS
                        for suffix in ("_median", "_iqr", "_delta", "_iqr_scaled_delta"))
FEATURE_COLUMNS = (BASE_COLUMNS + ("observed_prior_sessions", "history_below_minimum")
                   + HISTORY_COLUMNS)
KEY_COLUMNS = ("run_id", "contest_id", "handle")
ORDINARY_VERDICTS = {"OK", "WRONG_ANSWER", "COMPILATION_ERROR", "RUNTIME_ERROR",
                    "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED", "IDLENESS_LIMIT_EXCEEDED"}


def quantile(values, fraction):
    """Linear interpolation over sorted observations; empty means missing."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def problem_key(item):
    problem = item.get("problem", {})
    if not problem.get("index"):
        raise ValueError("Submission has no problem index")
    return (problem.get("contestId", item["contestId"]), problem["index"])


def summarize(items, contest):
    """One eligible official session. Never interpret timestamps as coding time."""
    sequence = sorted((x for x in items if x.get("contestId") == contest["id"]
                       and eligibility(x, contest)[0]),
                      key=lambda x: (x["creationTimeSeconds"], x["id"]))
    if not sequence:
        raise ValueError("No eligible official submissions for session")
    if len({x["id"] for x in sequence}) != len(sequence):
        raise ValueError("Duplicate submission IDs")
    grouped = defaultdict(list)
    for item in sequence:
        grouped[problem_key(item)].append(item)
    gaps = []
    switches = 0
    ratings = []
    for values in grouped.values():
        for previous, current in zip(values, values[1:]):
            gaps.append(current["creationTimeSeconds"] - previous["creationTimeSeconds"])
            switches += int(previous.get("programmingLanguage") != current.get("programmingLanguage"))
        observed = {x.get("problem", {}).get("rating") for x in values}
        numeric = {x for x in observed if isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x)}
        if len(numeric) > 1:
            raise ValueError("Conflicting problem difficulty snapshots")
        if numeric:
            ratings.append(next(iter(numeric)))
    duration = contest["durationSeconds"]
    if duration <= 0:
        raise ValueError("Contest duration must be positive")
    features = {
        "attempts": len(sequence), "attempted_problems": len(grouped), "retries": len(gaps),
        "retry_gap_min_seconds": min(gaps) if gaps else None,
        "retry_gap_p10_seconds": quantile(gaps, .1),
        "retry_gap_median_seconds": median(gaps) if gaps else None,
        "retry_fraction_le_15s": sum(x <= 15 for x in gaps) / len(gaps) if gaps else None,
        "retry_fraction_le_60s": sum(x <= 60 for x in gaps) / len(gaps) if gaps else None,
        "first_submission_fraction": (sequence[0]["creationTimeSeconds"] - contest["startTimeSeconds"]) / duration,
        "last_submission_fraction": (sequence[-1]["creationTimeSeconds"] - contest["startTimeSeconds"]) / duration,
        "attempts_per_hour": len(sequence) * 3600 / duration, "language_switches": switches,
        "problem_rating_mean": sum(ratings) / len(ratings) if ratings else None,
        "problem_rating_max": max(ratings) if ratings else None,
        "problem_rating_missing_fraction": 1 - len(ratings) / len(grouped),
    }
    # Exploratory diagnostics only. No administrative verdicts become features.
    ordinary = [x for x in sequence if x.get("verdict") in ORDINARY_VERDICTS]
    accepted = []
    for values in grouped.values():
        ok = [x for x in values if x.get("verdict") == "OK"]
        if ok:
            accepted.append(ok[0]["creationTimeSeconds"])
    accepted.sort()
    transitions = Counter()
    for values in grouped.values():
        for previous, current in zip(values, values[1:]):
            if previous.get("verdict") in ORDINARY_VERDICTS and current.get("verdict") in ORDINARY_VERDICTS:
                transitions[previous["verdict"] + "->" + current["verdict"]] += 1
    diagnostics = {
        "ordinary_verdict_records": len(ordinary),
        "other_or_missing_verdict_records": len(sequence) - len(ordinary),
        "accepted_problems_snapshot": len(accepted),
        "first_accept_spacing_median_seconds": median([b - a for a, b in zip(accepted, accepted[1:])]) if len(accepted) > 1 else None,
        "ordinary_verdict_counts": dict(sorted(Counter(x["verdict"] for x in ordinary).items())),
        "ordinary_verdict_transitions": dict(sorted(transitions.items())),
    }
    return features, diagnostics, sequence


def session_features(items, contests, target_id, panel_ids, lookback_days, expected_prior=None):
    """Pure builder: only the frozen panel strictly before this target contributes."""
    target = contests[target_id]
    start = target["startTimeSeconds"]
    earlier = {identifier: contests[identifier] for identifier in panel_ids
               if identifier in contests and identifier != target_id
               and start - lookback_days * 86400 <= contests[identifier]["startTimeSeconds"]
               and contests[identifier]["startTimeSeconds"] + contests[identifier]["durationSeconds"] < start}
    target_features, diagnostics, target_sequence = summarize(items, target)
    by_contest = defaultdict(list)
    for item in items:
        identifier = item.get("contestId")
        if identifier in earlier and eligibility(item, earlier[identifier])[0]:
            by_contest[identifier].append(item)
    prior = [summarize(by_contest[x], earlier[x])[0] for x in sorted(by_contest)]
    if expected_prior is not None and expected_prior != len(prior):
        raise ValueError("Normalized earlier sessions do not match panel coverage")
    features = target_features | {"observed_prior_sessions": len(prior),
                                  "history_below_minimum": int(len(prior) < MIN_HISTORY)}
    for name in BASE_COLUMNS:
        values = [x[name] for x in prior if x[name] is not None]
        sufficient = len(values) >= MIN_HISTORY
        center = median(values) if sufficient else None
        spread = quantile(values, .75) - quantile(values, .25) if sufficient else None
        delta = target_features[name] - center if center is not None and target_features[name] is not None else None
        features["prior_" + name + "_median"] = center
        features["prior_" + name + "_iqr"] = spread
        features["prior_" + name + "_delta"] = delta
        features["prior_" + name + "_iqr_scaled_delta"] = delta / spread if delta is not None and spread > 0 else None
    used = target_sequence + [item for identifier in sorted(by_contest) for item in
                              sorted(by_contest[identifier], key=lambda x: (x["creationTimeSeconds"], x["id"]))]
    return features, diagnostics, used


def csv_bytes(rows, columns):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_feature_report(store, run_id, output):
    audit = audit_run(store, run_id)
    if not audit["collection_checks_passed"]:
        raise ValueError("Collection audit must pass before building features")
    run = store.db.execute("SELECT config_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    config = json.loads(run[0])
    if config.get("sampling_strategy") != "contest":
        raise ValueError("This feature version supports frozen contest samples only")
    contests = {row[0]: json.loads(row[1]) for row in store.db.execute(
        "SELECT contest_id,payload_json FROM contests WHERE run_id=?", (run_id,))}
    panel_ids = [row[0] for row in store.db.execute(
        "SELECT contest_id FROM contest_panel WHERE run_id=? AND role='history'", (run_id,))]
    targets = list(store.db.execute("SELECT contest_id FROM contest_panel WHERE run_id=? AND role='target'", (run_id,)))
    target_id = targets[0][0]
    participants = list(store.db.execute(
        "SELECT p.* FROM participant_coverage p JOIN sessions s USING(run_id,contest_id,handle) "
        "WHERE p.run_id=? AND p.selected=1 AND s.eligible=1 ORDER BY p.handle COLLATE NOCASE", (run_id,)))
    by_handle = defaultdict(list)
    selected = {x["handle"].casefold() for x in participants}
    for row in store.db.execute("SELECT handle,payload_json FROM submissions WHERE run_id=?", (run_id,)):
        if row[0] and row[0].casefold() in selected:
            by_handle[row[0].casefold()].append(json.loads(row[1]))
    rows, diagnostics = [], []
    input_digest = hashlib.sha256()
    used_count = 0
    metadata = []
    for participant in participants:
        key = {"run_id": run_id, "contest_id": target_id, "handle": participant["handle"]}
        values, diagnostic, used = session_features(by_handle[participant["handle"].casefold()], contests,
            target_id, panel_ids, config["history_lookback_days"], participant["prior_contest_count"])
        rows.append(key | values)
        diagnostics.append(key | diagnostic)
        metadata.append(key | {"history_group": participant["history_group"],
                               "selection_probability": participant["selection_probability"],
                               "duration_seconds": contests[target_id]["durationSeconds"]})
        input_digest.update((canonical(key) + "\n" + canonical(used) + "\n").encode())
        used_count += len(used)
    if len(rows) != audit["population"]["development_sample_size"]:
        raise ValueError("Feature row count does not match development sample")
    panel = [contests[x] for x in sorted(set(panel_ids + [target_id]))]
    input_digest.update(canonical(panel).encode())
    artifacts = {
        "features.csv": csv_bytes(rows, KEY_COLUMNS + FEATURE_COLUMNS),
        "session_metadata.csv": csv_bytes(metadata, KEY_COLUMNS + ("history_group", "selection_probability", "duration_seconds")),
        "verdict_diagnostics.json": (json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n").encode(),
    }
    manifest = {
        "run_id": run_id, "feature_version": VERSION, "minimum_prior_values_per_feature": MIN_HISTORY,
        "rows": len(rows), "used_eligible_submissions": used_count,
        "model_columns": list(FEATURE_COLUMNS), "identifier_columns_not_predictors": list(KEY_COLUMNS),
        "input_sha256": input_digest.hexdigest(),
        "artifact_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in artifacts.items()},
        "missing_values": {name: sum(row[name] is None for row in rows) for name in FEATURE_COLUMNS},
        "history_groups": dict(Counter(x["history_group"] for x in metadata)),
        "training_ready": False,
        "notes": ["All target submissions are available post-contest. No suspicion labels are generated.",
                  "Retry intervals measure submission spacing, not coding time or feedback availability.",
                  "Difficulty is retrospective metadata. It is counted once per attempted problem.",
                  "Earlier context is restricted to eligible submissions in the frozen pre-target panel.",
                  "Blank CSV cells mean missing. Zero IQR leaves scaled deviations missing.",
                  "Verdict diagnostics are retrospective, exploratory and excluded from model_columns.",
                  "Identifiers and session_metadata are joins/provenance, not additional predictors.",
                  "Labels, evidence review, partitions and training-only preprocessing are still required."],
    }
    artifacts["feature_manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    output = Path(output)
    # Validate every existing artifact before writing any. Never silently replace a version.
    for name, body in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError(f"Frozen feature artifact differs: {path}. Use a new output/version.")
    output.mkdir(parents=True, exist_ok=True)
    for name, body in artifacts.items():
        path = output / name
        if not path.exists():
            path.write_bytes(body)
    return manifest
