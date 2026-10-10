"""Contest census, a frozen earlier-contest panel, and coverage-based sampling.

All raw pages are archived. Earlier submissions are normalized only for the
development sample, after coverage has been measured for the entire population.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import re

from .api import CollectionError
from .collect import collect_pages, require_list
from .sessions import eligibility
from .store import canonical


DEFAULTS = {
    "sampling_strategy": "contest", "selection_policy": "standard_div2_v1", "run_id": "contest-002", "target_contest_id": None,
    "contest_name_contains": "Div. 2", "min_contest_age_days": 7,
    "max_contest_age_days": 365, "history_lookback_days": 180,
    "history_contest_count": 20, "regular_min_contests": 5,
    "sample_size": 1000, "seed": 736,
    "sample_weights": {"regular": 0.7, "limited": 0.2, "none_observed": 0.1},
    "page_size": 10000, "max_contest_pages": 500,
    "request_interval_seconds": 2.1, "request_timeout_seconds": 60, "max_attempts": 4,
}


def validate_config(supplied):
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown contest configuration keys: {sorted(unknown)}")
    config = DEFAULTS | supplied
    if config["sampling_strategy"] != "contest":
        raise ValueError("sampling_strategy must be contest")
    if config["selection_policy"] != "standard_div2_v1":
        raise ValueError("Unsupported selection_policy")
    if not isinstance(config["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_id"]):
        raise ValueError("Invalid run_id")
    for name in ["history_contest_count", "regular_min_contests", "sample_size", "page_size",
                 "max_contest_pages", "history_lookback_days", "max_contest_age_days",
                 "request_timeout_seconds", "max_attempts"]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or type(config["min_contest_age_days"]) is not int:
        raise ValueError("seed and min_contest_age_days must be integers")
    if config["min_contest_age_days"] < 0 or config["max_contest_age_days"] < config["min_contest_age_days"]:
        raise ValueError("Invalid contest age bounds")
    if config["regular_min_contests"] > config["history_contest_count"] or config["page_size"] < 2:
        raise ValueError("Invalid history threshold or page size")
    if not isinstance(config["contest_name_contains"], str) or not config["contest_name_contains"].strip():
        raise ValueError("A non-empty contest name filter is required")
    target = config["target_contest_id"]
    if target is not None and (type(target) is not int or target <= 0):
        raise ValueError("target_contest_id must be null or a positive integer")
    interval = config["request_interval_seconds"]
    if type(interval) not in {int, float} or not math.isfinite(interval) or interval < 2.1:
        raise ValueError("Request interval must be finite and at least 2.1 seconds")
    weights = config["sample_weights"]
    if not isinstance(weights, dict) or set(weights) != {"regular", "limited", "none_observed"}:
        raise ValueError("sample_weights must specify all three coverage groups")
    if any(type(w) not in {int, float} or not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Sampling weights must be finite and positive")
    return config


def choose_panel(contests, config, as_of):
    def matching(item):
        return (item.get("phase") == "FINISHED" and item.get("type") == "CF"
                and isinstance(item.get("startTimeSeconds"), int)
                and isinstance(item.get("durationSeconds"), int) and item["durationSeconds"] > 0
                and config["contest_name_contains"].casefold() in item.get("name", "").casefold()
                and not re.search(r"div\s*\.?\s*1\b", item.get("name", ""), re.IGNORECASE)
                and "unrated" not in item.get("name", "").casefold())
    candidates = [item for item in contests if matching(item) and
                  config["min_contest_age_days"] <=
                  (as_of - item["startTimeSeconds"] - item["durationSeconds"]) / 86400
                  <= config["max_contest_age_days"]]
    candidates.sort(key=lambda x: (-x["startTimeSeconds"], -x["id"]))
    if config["target_contest_id"] is not None:
        candidates = [x for x in candidates if x["id"] == config["target_contest_id"]]
    if not candidates:
        raise CollectionError("No target matches the finished-contest, name and age requirements")
    target = candidates[0]
    start = target["startTimeSeconds"]
    earlier = [item for item in contests if matching(item) and item["id"] != target["id"]
               and start - config["history_lookback_days"] * 86400 <= item["startTimeSeconds"]
               and item["startTimeSeconds"] + item["durationSeconds"] < start]
    earlier.sort(key=lambda x: (-x["startTimeSeconds"], -x["id"]))
    panel = earlier[:config["history_contest_count"]]
    if len(panel) != config["history_contest_count"]:
        raise CollectionError("Insufficient comparable earlier contests in the lookback; use a new config/run rather than silently widening it")
    return target, panel


def stratified_sample(groups, size, seed, weights):
    """Weighted integer quotas, redistributing shortages without replacement."""
    pools = {name: sorted(set(handles), key=lambda x: (x.casefold(), x)) for name, handles in groups.items()}
    wanted = min(size, sum(map(len, pools.values())))
    quotas = {name: 0 for name in pools}
    remaining = wanted
    while remaining:
        available = [name for name in sorted(pools) if quotas[name] < len(pools[name])]
        total_weight = sum(weights[name] for name in available)
        shares = {name: remaining * weights[name] / total_weight for name in available}
        assigned = 0
        for name in available:
            take = min(len(pools[name]) - quotas[name], math.floor(shares[name]))
            quotas[name] += take
            assigned += take
        remaining -= assigned
        if not remaining:
            break
        for name in sorted(available, key=lambda x: (-(shares[x] % 1), x)):
            if remaining and quotas[name] < len(pools[name]):
                quotas[name] += 1
                remaining -= 1
    selected = set()
    for name, pool in pools.items():
        stable_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
        selected.update(random.Random(stable_seed).sample(pool, quotas[name]))
    probabilities = {name: quotas[name] / len(pool) if pool else 0.0 for name, pool in pools.items()}
    return selected, probabilities


def panel_complete(store, run_id):
    panel = list(store.db.execute("SELECT contest_id FROM contest_panel WHERE run_id=?", (run_id,)))
    if not panel:
        return False
    for row in panel:
        job = store.db.execute("SELECT status FROM jobs WHERE run_id=? AND job_key=?",
                               (run_id, f"contest:{row[0]}")).fetchone()
        if job is None or job[0] != "complete":
            return False
    return True


def build_contest_sessions(store, run_id, config):
    db = store.db
    target_row = db.execute("SELECT c.payload_json FROM contests c JOIN contest_panel p USING(run_id,contest_id) WHERE p.run_id=? AND p.role='target'", (run_id,)).fetchone()
    if target_row is None:
        return
    target = json.loads(target_row[0])
    target_id = target["id"]
    complete = panel_complete(store, run_id)
    records = defaultdict(list)
    for row in db.execute("SELECT payload_json FROM submissions WHERE run_id=? AND contest_id=? ORDER BY submission_id", (run_id, target_id)):
        item = json.loads(row[0])
        members = item.get("author", {}).get("members", [])
        if len(members) == 1 and members[0].get("handle"):
            records[members[0]["handle"].casefold()].append(item)
    prior = defaultdict(set)
    for row in db.execute("SELECT a.handle,a.contest_id FROM panel_participation a JOIN contest_panel p USING(run_id,contest_id) WHERE a.run_id=? AND p.role='history'", (run_id,)):
        prior[row["handle"].casefold()].add(row["contest_id"])
    in_standings = {row[0].casefold() for row in db.execute("SELECT handle FROM standings WHERE run_id=? AND contest_id=?", (run_id, target_id))}
    groups = {name: [] for name in config["sample_weights"]}
    for key, items in records.items():
        handle = items[0]["author"]["members"][0]["handle"]
        valid = [item for item in items if eligibility(item, target)[0]]
        reasons = sorted({eligibility(item, target)[1] for item in items if not eligibility(item, target)[0]})
        count = len(prior[key])
        db.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
                   (run_id, target_id, handle, int(bool(valid)), None if valid else ";".join(reasons),
                    len(valid), count, int(complete), int(key in in_standings)))
        db.execute("INSERT OR IGNORE INTO cohort VALUES(?,?,?,?,?)",
                   (run_id, handle, None, 0, f"https://codeforces.com/contest/{target_id}/status"))
        db.execute("INSERT OR IGNORE INTO labels(run_id,contest_id,handle) VALUES(?,?,?)", (run_id, target_id, handle))
        if valid and complete:
            group = "regular" if count >= config["regular_min_contests"] else "limited" if count else "none_observed"
            groups[group].append(handle)
    if complete:
        selected, probabilities = stratified_sample(groups, config["sample_size"], config["seed"], config["sample_weights"])
        selected_lower = {x.casefold() for x in selected}
        for group, handles in groups.items():
            for handle in handles:
                chosen = int(handle.casefold() in selected_lower)
                db.execute("INSERT OR REPLACE INTO participant_coverage VALUES(?,?,?,?,?,?,?)",
                           (run_id, target_id, handle, len(prior[handle.casefold()]), group, chosen, probabilities[group]))
                db.execute("UPDATE cohort SET selected=? WHERE run_id=? AND handle=?", (chosen, run_id, handle))
    db.commit()


def normalize_sample_history(store, run_id):
    db = store.db
    selected = {row[0].casefold() for row in db.execute("SELECT handle FROM participant_coverage WHERE run_id=? AND selected=1", (run_id,))}
    panel = {row["contest_id"]: json.loads(row["payload_json"]) for row in db.execute(
        "SELECT c.contest_id,c.payload_json FROM contests c JOIN contest_panel p USING(run_id,contest_id) WHERE p.run_id=? AND p.role='history'", (run_id,))}
    for row in db.execute("SELECT request_key,params_json FROM responses WHERE run_id=? ORDER BY request_key", (run_id,)).fetchall():
        params = json.loads(row["params_json"])
        contest_id = params.get("contestId")
        if contest_id not in panel or "from" not in params:
            continue
        items = json.loads(store.cached(run_id, row["request_key"]))["result"]
        retained = [item for item in items if eligibility(item, panel[contest_id])[0]
                    and item["author"]["members"][0]["handle"].casefold() in selected]
        store.ingest_submissions(run_id, row["request_key"], retained)


def collect_contest(client, config, progress=print):
    store, run_id = client.store, client.run_id
    db = store.db
    as_of = db.execute("SELECT created_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
    contests, _ = client.call("contest.list", gym="false")
    require_list(contests, "contest.list")
    target, history = choose_panel(contests, config, as_of)
    for position, item in enumerate([target] + history):
        db.execute("INSERT OR IGNORE INTO contests VALUES(?,?,?,?)", (run_id, item["id"], canonical(item), int(position == 0)))
        db.execute("INSERT OR IGNORE INTO contest_panel VALUES(?,?,?,?)", (run_id, item["id"], "target" if position == 0 else "history", position))
    db.commit()
    progress(f"Target {target['id']}: {target['name']}; {len(history)} earlier contests frozen")
    standings, _ = client.call("contest.standings", contestId=target["id"])
    if not isinstance(standings, dict) or not isinstance(standings.get("rows"), list) or standings.get("contest", {}).get("id") != target["id"]:
        raise CollectionError("Malformed or mismatched target standings")
    for row in standings["rows"]:
        for member in row.get("party", {}).get("members", []):
            db.execute("INSERT OR IGNORE INTO standings VALUES(?,?,?,?)", (run_id, target["id"], member["handle"], canonical(row)))
    db.commit()
    complete = collect_pages(client, f"contest:{target['id']}", "contest.status", {"contestId": target["id"]},
                             config["page_size"], config["max_contest_pages"], exhaust_on_empty=True, progress=progress)
    build_contest_sessions(store, run_id, config)
    if not complete:
        return False
    target_handles = {row[0].casefold() for row in db.execute("SELECT handle FROM sessions WHERE run_id=? AND eligible=1", (run_id,))}
    if not target_handles:
        raise CollectionError("Target has no eligible official individual submissions")
    for number, contest in enumerate(history, 1):
        progress(f"History panel {number}/{len(history)}: {contest['id']} {contest['name']}")
        def sink(items, key):
            # Idempotent participation facts; overlaps cannot increase counts.
            for item in items:
                if eligibility(item, contest)[0]:
                    handle = item["author"]["members"][0]["handle"]
                    if handle.casefold() in target_handles:
                        db.execute("INSERT OR IGNORE INTO panel_participation VALUES(?,?,?)", (run_id, contest["id"], handle))
            db.commit()
        complete &= collect_pages(client, f"contest:{contest['id']}", "contest.status", {"contestId": contest["id"]},
                                  config["page_size"], config["max_contest_pages"], item_sink=sink,
                                  exhaust_on_empty=True, progress=progress)
        if not complete:
            break
    build_contest_sessions(store, run_id, config)
    if complete:
        progress("Normalizing earlier submissions for the reproducible development sample from local archives")
        normalize_sample_history(store, run_id)
        counts = Counter(row[0] for row in db.execute("SELECT history_group FROM participant_coverage WHERE run_id=?", (run_id,)))
        progress(f"Historical coverage: {dict(counts)}")
    return complete
