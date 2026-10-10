"""Deterministic targeted pilot collection, with durable page checkpoints."""

from collections import Counter
import json
from pathlib import Path
import re

from .api import CollectionError
from .cohort import parse_organization_page, select_sample, validate_organization_url
from .store import canonical


DEFAULTS = {
    "run_id": "pilot-001", "organization_url": "https://codeforces.com/ratings/organization/5276",
    "sample_size": 24, "seed": 736, "contest_count": 3, "contest_ids": [],
    "contest_name_contains": "Div. 2", "min_contest_age_days": 7,
    "max_contest_age_days": 365, "history_lookback_days": 180, "page_size": 1000,
    "max_history_pages": 5, "max_session_pages": 10,
    "request_interval_seconds": 2.1, "request_timeout_seconds": 40, "max_attempts": 4,
}


def load_config(path: Path):
    supplied = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = DEFAULTS | supplied
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_id"]):
        raise ValueError("run_id must contain only letters, numbers, underscore or hyphen")
    validate_organization_url(config["organization_url"])
    for key in ["sample_size", "contest_count", "max_contest_age_days", "history_lookback_days",
                "page_size", "max_history_pages", "max_session_pages", "request_timeout_seconds", "max_attempts"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["page_size"] < 2:
        raise ValueError("page_size must be at least 2 for overlapping pagination")
    if config["request_interval_seconds"] < 2.1:
        raise ValueError("request_interval_seconds must be at least 2.1")
    if config["min_contest_age_days"] < 0 or config["max_contest_age_days"] < config["min_contest_age_days"]:
        raise ValueError("Invalid contest age bounds")
    if not isinstance(config["contest_ids"], list) or any(type(x) is not int or x <= 0 for x in config["contest_ids"]):
        raise ValueError("contest_ids must be a list of positive IDs")
    return config


def require_list(value, method):
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise CollectionError(f"{method} did not return a list of objects")
    return value


def choose_contests(contests, rating_events, config, as_of):
    """Pick recent, mature, similarly named rounds with cohort participation."""
    available = {item["id"]: item for item in contests}
    if config["contest_ids"]:
        selected = []
        for identifier in config["contest_ids"]:
            item = available.get(identifier)
            if not item or item.get("phase") != "FINISHED" or not item.get("startTimeSeconds"):
                raise CollectionError(f"Explicit contest {identifier} is absent or unfinished")
            if item["startTimeSeconds"] + item["durationSeconds"] > as_of:
                raise CollectionError(f"Explicit contest {identifier} ends in the future")
            selected.append(item)
        return sorted(selected, key=lambda x: x["startTimeSeconds"])
    votes = Counter(item["contestId"] for item in rating_events)
    candidates = []
    for item in contests:
        if item.get("phase") != "FINISHED" or item.get("type") != "CF" or not item.get("startTimeSeconds"):
            continue
        end = item["startTimeSeconds"] + item["durationSeconds"]
        age = (as_of - end) / 86400
        if not config["min_contest_age_days"] <= age <= config["max_contest_age_days"]:
            continue
        if config["contest_name_contains"].casefold() not in item.get("name", "").casefold():
            continue
        if votes[item["id"]]:
            candidates.append(item)
    candidates.sort(key=lambda x: (-votes[x["id"]], -x["startTimeSeconds"], x["id"]))
    selected = candidates[:config["contest_count"]]
    if len(selected) != config["contest_count"]:
        raise CollectionError("Insufficient matching completed contests; choose explicit contest_ids or a wider age range in a new run")
    return sorted(selected, key=lambda x: x["startTimeSeconds"])


def collect_pages(client, job_key, method, params, page_size, max_pages, cutoff=None):
    store, run_id = client.store, client.run_id
    job = store.job(run_id, job_key)
    if job["status"] == "complete":
        return True
    index, pages = job["next_from"], job["pages"]
    detail = json.loads(job["detail"] or "{}")
    previous_oldest = detail.get("oldest_id")
    while pages < max_pages:
        try:
            items, key = client.call(method, **params, **{"from": index, "count": page_size})
            require_list(items, method)
            for item in items:
                authors = {member.get("handle", "").casefold() for member in item.get("author", {}).get("members", [])}
                if params.get("handle", "").casefold() not in authors:
                    raise CollectionError("Submission author does not match the requested handle")
                if method == "contest.status" and item.get("contestId") != params["contestId"]:
                    raise CollectionError("Submission contest does not match the requested contest")
            ids = [item.get("id") for item in items]
            if any(type(identifier) is not int for identifier in ids):
                raise CollectionError("Malformed submission IDs")
            if any(left <= right for left, right in zip(ids, ids[1:])):
                raise CollectionError("Submission page is not in strictly decreasing ID order")
            if previous_oldest is not None and len(items) == page_size and ids[-1] >= previous_oldest:
                raise CollectionError("Pagination made no progress; possible repeated or shifted page")
            store.ingest_submissions(run_id, key, items)
            pages += 1
            complete = len(items) < page_size
            # user.status is documented as sorted by ID, not timestamp. Require
            # two consecutive old pages before stopping at a history cutoff.
            old_page = bool(items) and cutoff is not None and all(
                item["creationTimeSeconds"] < cutoff for item in items)
            consecutive_old = detail.get("old_pages", 0) + 1 if old_page else 0
            if consecutive_old >= 2:
                complete = True
            detail = {"oldest_id": ids[-1] if ids else previous_oldest,
                      "old_pages": consecutive_old, "cutoff": cutoff,
                      "stop_reason": "exhausted" if len(items) < page_size else "lookback_boundary" if complete else None}
            next_index = index + max(1, len(items) - 1)  # One-record overlap.
            store.update_job(run_id, job_key, next_index, "complete" if complete else "collecting",
                             pages, canonical(detail))
            if complete:
                return True
            previous_oldest, index = detail["oldest_id"], next_index
        except Exception as e:
            store.update_job(run_id, job_key, index, "failed", pages, canonical(detail))
            raise CollectionError(f"Job {job_key}: {e}") from e
    store.update_job(run_id, job_key, index, "capped", pages, canonical(detail))
    return False


def collect(client, config, progress=print):
    store, run_id = client.store, client.run_id
    as_of = store.db.execute("SELECT created_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
    html, _ = client.page(config["organization_url"])
    candidates, pagination = parse_organization_page(html, config["organization_url"])
    if pagination:
        raise CollectionError("Organisation page is paginated. Coverage requires implementing and testing those pages first.")
    sample = select_sample(candidates, config["sample_size"], config["seed"])
    selected_handles = {item.handle.casefold() for item in sample}
    for candidate in candidates:
        store.db.execute("INSERT OR IGNORE INTO cohort VALUES(?,?,?,?,?)",
                         (run_id, candidate.handle, candidate.rating,
                          int(candidate.handle.casefold() in selected_handles), config["organization_url"]))
    store.db.commit()
    progress(f"Organisation table: {len(candidates)} handles; deterministic sample: {len(sample)}")

    contests, _ = client.call("contest.list", gym="false")
    require_list(contests, "contest.list")
    for item in contests:
        store.db.execute("INSERT OR IGNORE INTO contests VALUES(?,?,?,0)",
                         (run_id, item["id"], canonical(item)))
    store.db.commit()

    all_ratings = []
    for number, candidate in enumerate(sample, 1):
        progress(f"Rating histories {number}/{len(sample)}: {candidate.handle}")
        events, _ = client.call("user.rating", handle=candidate.handle)
        require_list(events, "user.rating")
        for event in events:
            if event.get("handle", candidate.handle).casefold() != candidate.handle.casefold():
                raise CollectionError("Rating history handle mismatch")
            store.db.execute("INSERT OR IGNORE INTO rating_events VALUES(?,?,?,?)",
                             (run_id, candidate.handle, event["contestId"], canonical(event)))
        store.db.commit()
        all_ratings.extend(events)

    selected = choose_contests(contests, all_ratings, config, as_of)
    store.db.execute("UPDATE contests SET selected=0 WHERE run_id=?", (run_id,))
    for item in selected:
        store.db.execute("UPDATE contests SET selected=1 WHERE run_id=? AND contest_id=?", (run_id, item["id"]))
    store.db.commit()
    progress("Selected completed contests: " + ", ".join(f"{item['id']} {item['name']}" for item in selected))

    # Three whole public standings snapshots. API parameters deliberately contain
    # only contestId, as required for regular public contests.
    for contest in selected:
        progress(f"Standings: {contest['id']}")
        standings, _ = client.call("contest.standings", contestId=contest["id"])
        if not isinstance(standings, dict) or not isinstance(standings.get("rows"), list):
            raise CollectionError("Malformed standings response")
        if standings.get("contest", {}).get("id") != contest["id"]:
            raise CollectionError("Standings contest mismatch")
        for row in standings["rows"]:
            for member in row.get("party", {}).get("members", []):
                store.db.execute("INSERT OR IGNORE INTO standings VALUES(?,?,?,?)",
                                 (run_id, contest["id"], member["handle"], canonical(row)))
        store.db.commit()

    complete = True
    for number, candidate in enumerate(sample, 1):
        progress(f"Target submissions {number}/{len(sample)}: {candidate.handle}")
        # Query every sampled handle, even if absent from rating changes/standings.
        for contest in selected:
            key = f"session:{candidate.handle.casefold()}:{contest['id']}"
            complete &= collect_pages(client, key, "contest.status",
                                      {"contestId": contest["id"], "handle": candidate.handle},
                                      config["page_size"], config["max_session_pages"])
        cutoff = min(item["startTimeSeconds"] for item in selected) - config["history_lookback_days"] * 86400
        key = f"history:{candidate.handle.casefold()}"
        progress(f"Earlier submission archive: {candidate.handle}")
        complete &= collect_pages(client, key, "user.status", {"handle": candidate.handle},
                                  config["page_size"], config["max_history_pages"], cutoff)
    return complete
