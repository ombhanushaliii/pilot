"""Completed-contest API collection and frozen-model inference, without fitting."""
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .benchmark import FEATURE_COLUMNS, FEATURE_GROUPS, behavior_features, transform_features
from .vercel_portal import RemotePortal

CACHE = OrderedDict()
LOCK = Lock()
LAST_REQUEST = 0.
TTL = 600
PAGE_SIZE = 10000


class UpstreamError(ValueError):
    pass


def contest_id(value):
    text = str(value or "")
    if not text.isascii() or not text.isdigit() or not 1 <= int(text) <= 100000:
        raise ValueError("Enter the Codeforces contest ID from its /contest/ URL, not the round number.")
    return int(text)


def cf_request(method, params, deadline):
    global LAST_REQUEST
    delay = max(0., 2.1 - (time.monotonic() - LAST_REQUEST))
    if deadline - time.monotonic() < delay + 3:
        raise UpstreamError("Codeforces collection timed out. No partial contest was used. Try again shortly.")
    time.sleep(delay)
    LAST_REQUEST = time.monotonic()
    request = Request("https://codeforces.com/api/" + method + "?" + urlencode(params),
                      headers={"User-Agent": "PilotContestReview/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=min(15, deadline - time.monotonic())) as response:
            value = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        raise UpstreamError("Codeforces could not be reached. No saved contest was substituted. Try again shortly.") from error
    if value.get("status") != "OK":
        raise UpstreamError("Codeforces API: " + str(value.get("comment", "request failed"))[:200])
    return value["result"]


def prepare_contest(payload, contest, submissions):
    if contest.get("phase") != "FINISHED":
        raise ValueError("Post-contest review is available only after the contest has finished.")
    start, duration = contest.get("startTimeSeconds"), contest.get("durationSeconds")
    if type(start) is not int or type(duration) is not int or duration <= 0:
        raise ValueError("Contest timing is unavailable.")
    if start < 1735689600:
        raise ValueError("Choose a completed contest from 2025 onward. Round 1127 uses contest ID 2274.")
    grouped, seen = defaultdict(list), set()
    for item in submissions:
        sid = item.get("id")
        if sid in seen:
            raise UpstreamError("Submission pages changed during collection. Reload the contest to avoid incomplete sequences.")
        seen.add(sid)
        author = item.get("author", {})
        members = author.get("members", [])
        if (item.get("contestId") != contest["id"] or
            not start <= item.get("creationTimeSeconds", 0) < start + duration or
            author.get("participantType") not in {"CONTESTANT", "OUT_OF_COMPETITION"} or
            len(members) != 1 or author.get("ghost") or author.get("teamId") is not None):
            continue
        grouped[members[0]["handle"]].append(item)
    pre = json.loads((Path(payload) / "preprocessor.json").read_text(encoding="utf-8"))
    directory, eligible, vectors = [], [], []
    for handle, items in sorted(grouped.items(), key=lambda pair: pair[0].casefold()):
        items.sort(key=lambda s: (s["creationTimeSeconds"], s["id"]))
        features = behavior_features(items, contest)
        row = {"id": f"{contest['id']}:{handle}", "handle": handle,
               "attempts": len(items), "problem_count": features["attempted_problems"],
               "last_seconds": items[-1]["creationTimeSeconds"] - start,
               "eligible": features["attempted_problems"] >= 2,
               "submissions": [{"id": item["id"], "problem": item["problem"]["index"],
                                "problem_name": item["problem"].get("name", ""),
                                "language": item.get("programmingLanguage") or "Unknown",
                                "verdict": item.get("verdict") or "UNAVAILABLE",
                                "seconds": item["creationTimeSeconds"] - start} for item in items]}
        directory.append(row)
        if row["eligible"]:
            eligible.append(row)
            vectors.append(transform_features(features, pre))
    if not directory:
        raise ValueError("No individual in-contest submission sessions were found.")
    digest = hashlib.sha256(json.dumps(directory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model = json.loads((Path(payload) / "model.json").read_text(encoding="utf-8"))["selected"]
    snapshot = {"contest": contest, "snapshot_time": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": digest, "model": model, "rows": eligible,
                "feature_columns": FEATURE_COLUMNS, "feature_groups": FEATURE_GROUPS}
    portal = RemotePortal(payload, snapshot, np.asarray(vectors, dtype=np.float32).reshape((-1, 48)))
    portal.directory = directory
    portal.by_id = {row["id"]: row for row in directory}
    return portal


def get_live_portal(payload, value):
    cid = contest_id(value)
    with LOCK:
        now = time.monotonic()
        if cid in CACHE and now - CACHE[cid][0] < TTL:
            CACHE.move_to_end(cid)
            return CACHE[cid][1]
        deadline = now + 43
        contest = cf_request("contest.standings", {"contestId": cid}, deadline)["contest"]
        if contest.get("phase") != "FINISHED":
            raise ValueError("Post-contest review is available only after the contest has finished.")
        if contest.get("startTimeSeconds", 0) < 1735689600:
            raise ValueError("Choose a completed contest from 2025 onward. Round 1127 uses contest ID 2274.")
        submissions = []
        for page in range(15):
            batch = cf_request("contest.status", {"contestId": cid, "from": page * PAGE_SIZE + 1,
                               "count": PAGE_SIZE}, deadline)
            submissions.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
        else:
            raise UpstreamError("This contest exceeds the demo collection limit. No partial contest was used.")
        portal = prepare_contest(payload, contest, submissions)
        CACHE[cid] = (time.monotonic(), portal)
        CACHE.move_to_end(cid)
        while len(CACHE) > 3:
            CACHE.popitem(last=False)
        return portal
