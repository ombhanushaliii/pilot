"""Session eligibility and strictly pre-target historical context."""

import json


def eligibility(submission, contest):
    author = submission.get("author", {})
    if author.get("ghost"):
        return False, "ghost"
    if len(author.get("members", [])) != 1 or author.get("teamId") is not None:
        return False, "team_or_missing_author"
    kind = author.get("participantType")
    if kind == "OUT_OF_COMPETITION":
        return False, "out_of_competition_requires_manual_eligibility_review"
    if kind != "CONTESTANT":
        return False, f"participant_type:{kind or 'missing'}"
    start = contest["startTimeSeconds"]
    # Half-open interval avoids including post-contest activity at the end boundary.
    if not start <= submission["creationTimeSeconds"] < start + contest["durationSeconds"]:
        return False, "outside_contest_window"
    return True, None


def prior_sessions(submissions, contests, target, lookback_days):
    """Returns full earlier eligible contest sessions inside the lookback."""
    start = target["startTimeSeconds"]
    cutoff = start - lookback_days * 86400
    identifiers = set()
    for submission in submissions:
        earlier = contests.get(submission.get("contestId"))
        if not earlier or earlier["id"] == target["id"] or not earlier.get("startTimeSeconds"):
            continue
        if earlier["startTimeSeconds"] + earlier["durationSeconds"] >= start:
            continue
        if not cutoff <= earlier["startTimeSeconds"] < start:
            continue
        if eligibility(submission, earlier)[0]:
            identifiers.add(earlier["id"])
    return sorted(identifiers, key=lambda x: contests[x]["startTimeSeconds"], reverse=True)


def build_sessions(store, run_id, lookback_days):
    contests = {row["contest_id"]: json.loads(row["payload_json"]) for row in store.db.execute(
        "SELECT * FROM contests WHERE run_id=?", (run_id,))}
    selected = [row[0] for row in store.db.execute(
        "SELECT contest_id FROM contests WHERE run_id=? AND selected=1", (run_id,))]
    handles = [row[0] for row in store.db.execute(
        "SELECT handle FROM cohort WHERE run_id=? AND selected=1", (run_id,))]
    for handle in handles:
        submissions = [json.loads(row[0]) for row in store.db.execute(
            "SELECT payload_json FROM submissions WHERE run_id=? AND handle=? ORDER BY created_at,submission_id",
            (run_id, handle))]
        history = store.db.execute("SELECT status FROM jobs WHERE run_id=? AND job_key=?",
                                   (run_id, "history:" + handle.casefold())).fetchone()
        for contest_id in selected:
            contest = contests[contest_id]
            matching = [item for item in submissions if item.get("contestId") == contest_id]
            valid = [item for item in matching if eligibility(item, contest)[0]]
            reasons = sorted({eligibility(item, contest)[1] for item in matching if not eligibility(item, contest)[0]})
            reason = None if valid else ";".join(reasons) if matching else "no_submissions_returned"
            in_standings = store.db.execute(
                "SELECT 1 FROM standings WHERE run_id=? AND contest_id=? AND handle=?",
                (run_id, contest_id, handle)).fetchone() is not None
            previous = prior_sessions(submissions, contests, contest, lookback_days)
            store.db.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
                             (run_id, contest_id, handle, int(bool(valid)), reason, len(valid), len(previous),
                              int(history is not None and history[0] == "complete"), int(in_standings)))
            # Preserve any manually entered labels when rebuilding sessions.
            store.db.execute("INSERT OR IGNORE INTO labels(run_id,contest_id,handle) VALUES(?,?,?)",
                             (run_id, contest_id, handle))
    store.db.commit()
