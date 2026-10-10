"""Integrity checks for the population and fixed history panel."""

from collections import Counter, defaultdict
import json

from .contest_collect import stratified_sample
from .sessions import eligibility


def audit_population(store, run_id, config, issues):
    db = store.db
    panel = [{**dict(row), "contest": json.loads(row["payload_json"])} for row in db.execute(
        "SELECT p.contest_id,p.role,p.position,c.payload_json FROM contest_panel p JOIN contests c USING(run_id,contest_id) WHERE p.run_id=? ORDER BY p.position", (run_id,))]
    targets = [row for row in panel if row["role"] == "target"]
    histories = [row for row in panel if row["role"] == "history"]
    if len(targets) != 1 or len(histories) != config["history_contest_count"]:
        issues.append({"code": "panel_selection_incomplete", "targets": len(targets), "history": len(histories)})
    if targets:
        target = targets[0]["contest"]
        start = target["startTimeSeconds"]
        for row in histories:
            contest = row["contest"]
            if (contest["startTimeSeconds"] + contest["durationSeconds"] >= start
                    or contest["startTimeSeconds"] < start - config["history_lookback_days"] * 86400):
                issues.append({"code": "history_panel_time", "contest_id": contest["id"]})
        job = db.execute("SELECT detail,status FROM jobs WHERE run_id=? AND job_key=?", (run_id, f"contest:{target['id']}")).fetchone()
        target_rows = db.execute("SELECT count(*) FROM submissions WHERE run_id=? AND contest_id=?", (run_id, target["id"])).fetchone()[0]
        if job and job["status"] == "complete" and target_rows != json.loads(job["detail"])["unique_items"]:
            issues.append({"code": "target_submission_census_mismatch", "normalized": target_rows})
    coverage = list(db.execute("SELECT * FROM participant_coverage WHERE run_id=?", (run_id,)))
    eligible = {row[0].casefold(): row[1] for row in db.execute("SELECT handle,prior_contest_count FROM sessions WHERE run_id=? AND eligible=1", (run_id,))}
    if {row["handle"].casefold() for row in coverage} != set(eligible):
        issues.append({"code": "coverage_population_mismatch", "eligible": len(eligible), "coverage": len(coverage)})
    groups = {name: [] for name in config["sample_weights"]}
    for row in coverage:
        handle = row["handle"].casefold()
        count = db.execute("SELECT count(*) FROM panel_participation a JOIN contest_panel p USING(run_id,contest_id) WHERE a.run_id=? AND a.handle=? AND p.role='history'", (run_id, handle)).fetchone()[0]
        group = "regular" if count >= config["regular_min_contests"] else "limited" if count else "none_observed"
        if count != row["prior_contest_count"] or count != eligible.get(handle) or group != row["history_group"]:
            issues.append({"code": "coverage_count_or_group_mismatch", "handle": row["handle"]})
        groups[group].append(row["handle"])
    expected, probabilities = stratified_sample(groups, config["sample_size"], config["seed"], config["sample_weights"])
    actual = {row["handle"] for row in coverage if row["selected"]}
    if expected != actual:
        issues.append({"code": "development_sample_mismatch", "expected_size": len(expected), "actual_size": len(actual)})
    selected_lower = {x.casefold() for x in actual}
    cohort_selected = {row[0].casefold() for row in db.execute("SELECT handle FROM cohort WHERE run_id=? AND selected=1", (run_id,))}
    if cohort_selected != selected_lower:
        issues.append({"code": "cohort_sample_mismatch"})
    if any(abs(row["selection_probability"] - probabilities[row["history_group"]]) > 1e-12 for row in coverage):
        issues.append({"code": "sampling_probability_mismatch"})
    history_contests = {row["contest_id"]: row["contest"] for row in histories}
    normalized_prior = defaultdict(set)
    for row in db.execute("SELECT handle,contest_id,payload_json FROM submissions WHERE run_id=?", (run_id,)):
        if targets and row["contest_id"] == targets[0]["contest_id"]:
            continue
        contest = history_contests.get(row["contest_id"])
        handle = (row["handle"] or "").casefold()
        if contest is None or handle not in selected_lower or not eligibility(json.loads(row["payload_json"]), contest)[0]:
            issues.append({"code": "normalized_history_outside_sample_or_panel", "contest_id": row["contest_id"]})
            break
        normalized_prior[handle].add(row["contest_id"])
    missing_history = sum(len(normalized_prior[x["handle"].casefold()]) != x["prior_contest_count"] for x in coverage if x["selected"])
    if missing_history:
        issues.append({"code": "sample_history_normalization_incomplete", "participants": missing_history})
    return {
        "panel": [{key: value for key, value in row.items() if key != "payload_json"} for row in panel],
        "eligible_population": len(eligible), "coverage_groups": dict(Counter(x["history_group"] for x in coverage)),
        "development_sample_size": len(actual),
        "development_sample_groups": dict(Counter(x["history_group"] for x in coverage if x["selected"])),
        "sampling_note": "Weighted random sample by observed panel participation, not by misconduct. Group inclusion probabilities are recorded. College and country are not collected.",
        "coverage_note": "none_observed means no eligible participation in this panel; it does not establish a new account or fair participation.",
    }
