"""Lock tabular RL references on development data, then score the fresh holdout."""
import argparse
from bisect import bisect_left
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import FEATURE_COLUMNS, file_hash, write_json
from cf_pilot.rl import STATE_FEATURES, TabularAgent, rollout
from cf_pilot.rl_data import binary_metrics


def rows(split):
    folder = ROOT / "data/deep-benchmark-2026-001"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    pre = json.loads((ROOT / "data/benchmark-2026-001/preprocessor.json").read_text(encoding="utf-8"))
    path, targets = folder / split / "features.csv", folder / split / "targets.csv"
    if file_hash(path) != summary["partitions"][split]["artifacts"]["features.csv"] or file_hash(targets) != summary["partitions"][split]["artifacts"]["targets.csv"]:
        raise ValueError("Legacy evaluation input changed")
    bins, y = [], []
    with path.open(encoding="utf-8", newline="") as f, targets.open(encoding="utf-8", newline="") as t:
        for feature, target in zip(csv.DictReader(f), csv.DictReader(t)):
            if feature["session_id"] != target["session_id"]:
                raise ValueError("Legacy session mismatch")
            bins.append(tuple(bisect_left(pre["stats"][name]["quartiles"], float(feature[name])) if feature[name] else 4 for name in STATE_FEATURES))
            y.append(int(target["recorded_outcome"]))
    return bins, np.array(y)


def evaluate(split):
    bins, y = rows(split)
    measured, hashes = {}, {}
    old = ROOT / "data/training-2026-002"
    for algorithm in ("q_learning", "sarsa"):
        for seed in (42,736,2026,13,2718):
            name = f"{algorithm}_{seed}"
            path = old / (name + ".json")
            saved = json.loads(path.read_text(encoding="utf-8"))
            agent, cache = TabularAgent.restore(saved["model"]), {}
            p, counts = [], []
            for row in bins:
                if row not in cache: cache[row] = rollout(row, agent)
                prediction, inspected, _ = cache[row]
                p.append(prediction); counts.append(inspected)
            measured[name], hashes[name] = binary_metrics(y, p, counts), file_hash(path)
    return measured, hashes


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["select","evaluate"]); args=parser.parse_args()
    report=ROOT / "reports/deep-rl-2026-001"; report.mkdir(parents=True,exist_ok=True)
    if args.mode=="select":
        measured, hashes=evaluate("validation")
        selected={a:max([n for n in measured if n.startswith(a)],key=lambda n:(measured[n]["f1"],measured[n]["precision"])) for a in ("q_learning","sarsa")}
        write_json(report / "legacy-rl-selection.json",{"selected":selected,"validation":measured,"model_sha256":hashes,"fresh_features_opened":False})
        print(json.dumps({"selected":selected,"validation":{a:measured[n] for a,n in selected.items()}},indent=2))
    else:
        frozen=json.loads((report / "legacy-rl-selection.json").read_text())
        measured,hashes=evaluate("test")
        if hashes!=frozen["model_sha256"]: raise ValueError("Legacy policies changed after selection")
        aggregate={a:{field:{"mean":float(np.mean([v[field] for n,v in measured.items() if n.startswith(a)])),"std":float(np.std([v[field] for n,v in measured.items() if n.startswith(a)],ddof=1))} for field in ["precision","recall","f1","average_inspections"]} for a in ("q_learning","sarsa")}
        write_json(report / "legacy-rl-fresh-test.json",{"fresh_test":measured,"seed_aggregate":aggregate,"selected_on_validation":frozen["selected"],"representation":"six historical train-binned summaries; different inputs and training data from deep models"})
        print(json.dumps({"legacy_rl_aggregate":aggregate},indent=2))
