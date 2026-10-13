"""RL-only model selection on validation, followed by a single fresh final test."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json
from cf_pilot.deep_rl import QNetwork, infer, learn
from cf_pilot.rl_data import binary_metrics, load_data
from cf_pilot.store import canonical


def run():
    config = json.loads((ROOT / "config/deep-rl.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = ROOT / "data" / config["benchmark_id"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    pre = json.loads((folder / "preprocessor.json").read_text(encoding="utf-8"))
    if not summary["complete"] or file_hash(folder / "preprocessor.json") != summary["preprocessor_sha256"]:
        raise ValueError("Expanded benchmark incomplete or changed")
    if pre["training_features_sha256"] != summary["partitions"]["train"]["artifacts"]["features.csv"]:
        raise ValueError("Expanded preprocessing provenance mismatch")
    models, report = ROOT / "data" / config["run_id"], ROOT / "reports" / config["run_id"]
    models.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "benchmark_sha256": file_hash(folder / "summary.json"),
                "preprocessor_sha256": file_hash(folder / "preprocessor.json"),
                "code_sha256": {p: file_hash(ROOT / p) for p in ["cf_pilot/deep_rl.py", "cf_pilot/rl_data.py", "tools/train_deep_rl.py"]},
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
                "selection_metric": "validation F1, precision breaks ties", "uses_supervised_algorithms": False,
                "target_is_misconduct_truth": False, "previous_test_is_development_data": True}
    frozen = models / "manifest.json"
    if frozen.exists() and canonical(json.loads(frozen.read_text(encoding="utf-8"))) != canonical(manifest):
        raise ValueError("Deep RL run changed; use a new run ID")
    write_json(frozen, manifest)
    if (models / "complete.json").exists():
        for relative, sha in json.loads((models / "complete.json").read_text())["artifacts"].items():
            if file_hash(ROOT / relative) != sha:
                raise ValueError("Deep RL artifact changed")
        print("Verified completed RL-only run; no retraining or retuning", flush=True)
        return
    write_json(models / "status.json", {"stage": "training"})
    print("Loading expanded training/validation; fresh final-test feature tables remain unopened", flush=True)
    train, val = [load_data(folder, p, summary, pre) for p in ("train", "validation")]
    if set(train["keys"]) & set(val["keys"]):
        raise ValueError("Deep RL partitions overlap")
    results = {}

    def fit(candidate, seed):
        name = candidate["name"] + "_" + str(seed)
        path = models / (name + ".pt")
        meta_path = models / (name + ".json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if file_hash(path) != meta["model_sha256"]:
                raise ValueError("Cached neural model changed")
        else:
            started = time.monotonic()
            saved = learn(train, val, candidate, config["settings"], seed, binary_metrics)
            weights = saved.pop("weights")
            torch.save(weights, path)
            meta = {**saved, "model_sha256": file_hash(path), "elapsed_seconds": round(time.monotonic() - started, 2)}
            write_json(meta_path, meta)
        results[name] = meta
        return name

    for candidate in config["candidates"]:
        fit(candidate, config["search_seed"])
    best = max(results, key=lambda name: (results[name]["validation"]["f1"], results[name]["validation"]["precision"]))
    selected_candidate = results[best]["candidate"]
    for seed in config["repeat_seeds"]:
        fit(selected_candidate, seed)
    selected = max(results, key=lambda name: (results[name]["validation"]["f1"], results[name]["validation"]["precision"]))
    selection = {"selected": selected, "selected_candidate": results[selected]["candidate"],
                 "selected_update": results[selected]["selected_update"],
                 "metric": "validation F1", "validation": {n: v["validation"] for n, v in results.items()},
                 "model_sha256": {n: v["model_sha256"] for n, v in results.items()}}
    write_json(models / "selection.json", selection)
    write_json(models / "status.json", {"stage": "fresh_test", "selection_sha256": file_hash(models / "selection.json")})
    print("RL-only selection frozen: " + selected + ". Opening fresh contest 2274 for final evaluation", flush=True)
    test = load_data(folder, "test", summary, pre)
    if set(test["keys"]) & (set(train["keys"]) | set(val["keys"])):
        raise ValueError("Fresh-test session overlap")
    training_handles = {m["handle"] for m in train["metadata"]}
    seen_handles = training_handles | {m["handle"] for m in val["metadata"]}
    measurements, predictions = {}, None
    for name, meta in results.items():
        candidate = meta["candidate"]
        network = QNetwork(candidate["width"], candidate["dueling"])
        network.load_state_dict(torch.load(models / (name + ".pt"), map_location="cpu", weights_only=True))
        pred, counts, masks, paths = infer(network, test["x"], traces=name == selected)
        slices = {"all": binary_metrics(test["y"], pred, counts)}
        conditions = {"unseen_in_train_and_validation": np.array([m["handle"] not in seen_handles for m in test["metadata"]])}
        for stratum in sorted({m["participation_stratum"] for m in test["metadata"]}):
            conditions["participation:" + stratum] = np.array([m["participation_stratum"] == stratum for m in test["metadata"]])
        for key, condition in conditions.items():
            if condition.any(): slices[key] = binary_metrics(test["y"][condition], pred[condition], counts[condition])
        measurements[name] = slices
        if name == selected:
            predictions = (pred, counts, masks, paths)
    pred, counts, masks, paths = predictions
    action_names = ["inspect_pacing", "inspect_trajectory", "inspect_language", "low_priority", "review"]
    examples = {}
    with (report / "fresh-test-predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session_id", "handle", "contest_id", "participation_stratum",
            "recommendation", "recorded_outcome", "inspections", "actions"])
        writer.writeheader()
        for i, identity in enumerate(test["metadata"]):
            path = [action_names[a] for a in paths[i]]
            row = {**identity, "recommendation": "REVIEW" if pred[i] else "LOW_PRIORITY",
                   "recorded_outcome": int(test["y"][i]), "inspections": int(counts[i]), "actions": " > ".join(path)}
            writer.writerow(row)
            kind = ("true_positive" if pred[i] else "false_negative") if test["y"][i] else ("false_positive" if pred[i] else "true_negative")
            if kind not in examples:
                examples[kind] = {"example_id": "fresh_" + str(i), "recommendation": row["recommendation"],
                                  "recorded_outcome": row["recorded_outcome"], "actions": path, "inspections": row["inspections"]}
    repeat_names = [name for name, meta in results.items() if meta["candidate"]["name"] == selected_candidate["name"]]
    aggregate = {metric: {"mean": float(np.mean([measurements[n]["all"][metric] for n in repeat_names])),
                         "std": float(np.std([measurements[n]["all"][metric] for n in repeat_names], ddof=1))}
                 for metric in ["precision", "recall", "f1", "average_inspections"]}
    output = {"run_id": config["run_id"], "complete": True, "selected": selected, "selection": selection,
              "benchmark_id": config["benchmark_id"], "validation": {n: v["validation"] for n, v in results.items()},
              "fresh_test": measurements, "repeat_candidate": selected_candidate["name"], "repeat_seed_aggregate": aggregate,
              "training_sessions": len(train["y"]), "validation_sessions": len(val["y"]),
              "test_sessions": len(test["y"]), "all_algorithms_are_rl": True,
              "fresh_test_opened_after_selection": True, "fresh_verdict_stability": "recent finished snapshot; skips may change later",
              "interpretation": "Recorded-outcome review policy; not verified cheating or AI-use accuracy"}
    write_json(report / "results.json", output)
    write_json(report / "learning-curves.json", {n: v["history"] for n, v in results.items()})
    write_json(report / "demo-cases.json", {"selected": selected, "cases": examples})
    write_json(models / "status.json", {"stage": "complete"})
    artifacts = {str(p.relative_to(ROOT)): file_hash(p) for p in list(models.glob("*")) + list(report.glob("*")) if p.is_file() and p.name != "complete.json"}
    write_json(models / "complete.json", {"artifacts": artifacts})
    print(json.dumps({"selected": selected, "validation": results[selected]["validation"], "fresh_test": measurements[selected]}, indent=2), flush=True)


if __name__ == "__main__":
    run()
