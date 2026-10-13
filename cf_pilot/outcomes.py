"""Describe recorded SKIPPED patterns without assigning misconduct labels."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib

from .store import canonical


YEAR_START = 1735689600
YEAR_END = 1767225600
UNSTABLE = {None, "SUBMITTED", "TESTING"}


def year_bounds(year):
    if type(year) is not int or not 1970 <= year <= 9998:
        raise ValueError("Invalid UTC study year")
    return (int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()),
            int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()))


def audit_outcomes(submissions, contest, complete, *, year=2025, on_session=None):
    """Audit individual in-window sessions, preserving uncertain participation.

    Replacement-compatible means only that a later non-skipped attempt exists.
    Neither these descriptions nor participation type establish a skip's cause.
    """
    year_start, year_end = year_bounds(year)
    start, duration = contest["startTimeSeconds"], contest["durationSeconds"]
    if (type(start) is not int or type(duration) is not int or duration <= 0
            or not year_start <= start < start + duration <= year_end):
        raise ValueError(f"Contest must lie entirely within UTC {year}")
    sessions, excluded, seen = defaultdict(list), Counter(), {}
    for sub in submissions:
        sid, created = sub.get("id"), sub.get("creationTimeSeconds")
        if type(sid) is not int or type(created) is not int:
            raise ValueError("Invalid submission ID or timestamp")
        if sub.get("contestId") != contest["id"]:
            raise ValueError("Submission contest mismatch")
        digest = hashlib.sha256(canonical(sub).encode("utf-8")).digest()
        if sid in seen:
            if seen[sid] != digest:
                raise ValueError("Conflicting duplicate submission")
            continue
        seen[sid] = digest
        if not year_start <= created < year_end:
            excluded[f"outside_{year}"] += 1
            continue
        if not start <= created < start + duration:
            excluded["outside_contest_window"] += 1
            continue
        author = sub.get("author", {})
        members = author.get("members", [])
        if author.get("ghost") or author.get("teamId") is not None or len(members) != 1:
            excluded["ghost_team_or_missing_author"] += 1
            continue
        handle = members[0].get("handle")
        if not isinstance(handle, str) or not handle:
            raise ValueError("Invalid individual handle")
        kind = author.get("participantType")
        if kind not in {"CONTESTANT", "OUT_OF_COMPETITION"}:
            excluded["participant_type:" + str(kind)] += 1
            continue
        index = sub.get("problem", {}).get("index")
        if not isinstance(index, str) or not index:
            raise ValueError("Missing problem identity")
        sessions[handle.casefold()].append(sub)

    records = []
    for handle, items in sorted(sessions.items()):
        if on_session is not None:
            on_session(handle, items)
        by_problem = defaultdict(list)
        for sub in items:
            by_problem[sub["problem"]["index"]].append(sub)
        skipped = sum(s.get("verdict") == "SKIPPED" for s in items)
        final_skips = replacement_compatible = 0
        for attempts in by_problem.values():
            attempts.sort(key=lambda s: (s["creationTimeSeconds"], s["id"]))
            final_skips += attempts[-1].get("verdict") == "SKIPPED"
            later_non_skipped = False
            for sub in reversed(attempts):
                verdict = sub.get("verdict")
                if verdict == "SKIPPED" and later_non_skipped:
                    replacement_compatible += 1
                elif verdict not in UNSTABLE and verdict != "SKIPPED":
                    later_non_skipped = True
        unstable = any(s.get("verdict") in UNSTABLE for s in items)
        if not complete or unstable:
            pattern = "outcome_unresolved"
        elif not skipped:
            pattern = "no_skipped_observed"
        elif skipped == len(items):
            pattern = ("all_skipped_multiple_problems" if len(by_problem) >= 2
                       else "all_skipped_single_problem")
        elif final_skips:
            pattern = "some_final_attempts_skipped"
        else:
            pattern = "skips_with_later_non_skipped_attempts"
        kinds = sorted({s["author"]["participantType"] for s in items})
        records.append({
            "handle": handle, "contest_id": contest["id"],
            "participation_stratum": kinds[0] if len(kinds) == 1 else "mixed",
            "submission_count": len(items), "problem_count": len(by_problem),
            "skipped_count": skipped, "final_attempts_skipped": final_skips,
            "replacement_compatible_skips": replacement_compatible,
            "pattern": pattern, "misconduct_label": "unknown",
            "observed_skipped": None if not complete or unstable else int(skipped > 0),
        })
    strata = defaultdict(Counter)
    for row in records:
        strata[row["participation_stratum"]][row["pattern"]] += 1
    summary = {
        "contest_id": contest["id"], "contest_name": contest["name"],
        "collection_complete": bool(complete), "raw_unique_submissions": len(seen),
        "in_window_individual_submissions": sum(r["submission_count"] for r in records),
        "sessions": len(records), "excluded_submissions": dict(excluded),
        "patterns_by_participation": {k: dict(v) for k, v in sorted(strata.items())},
        "misconduct_labels_assigned": 0,
        "interpretation": "Observed verdict patterns only; no reason or AI-use inference.",
    }
    return summary, records
