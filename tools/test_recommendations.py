"""Frozen RL recommendation ablations over two cached chronological development folds."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import argparse
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import sys
import shutil
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json
from cf_pilot.context import CONTEXT_COLUMNS
from cf_pilot.recommendation_rl import Network, infer, learn, measure, observations, NFEATURES
from cf_pilot.store import canonical


def frozen(path, value):
    if path.exists() and canonical(json.loads(path.read_text(encoding="utf-8"))) != canonical(value):
        raise ValueError("Frozen ablation changed: " + str(path))
    if not path.exists():
        write_json(path, value)


def load(ids, complete):
    n = sum(complete["contests"][str(cid)]["rows"] for cid in ids)
    raw, y, cid_vector = np.empty((n, NFEATURES)), np.empty(n, dtype=np.int8), np.empty(n, dtype=np.int32)
    keys, metadata, seen, i = [], [], set(), 0
    for cid in ids:
        extra = ROOT / "data/recommendation-tests-001/contests" / str(cid)
        old = ROOT / "data/deep-benchmark-2026-001/contests" / str(cid)
        manifest = complete["contests"][str(cid)]
        paths = [extra / "features.csv", old / "targets.csv", old / "metadata.csv"]
        for path, wanted in zip(paths, [manifest["sha256"], manifest["target_sha256"], manifest["metadata_sha256"]]):
            if file_hash(path) != wanted:
                raise ValueError("Ablation input checksum changed")
        opened = [p.open(encoding="utf-8", newline="") for p in paths]
        try:
            readers = [csv.DictReader(s) for s in opened]
            if readers[0].fieldnames != ["session_id", *CONTEXT_COLUMNS]:
                raise ValueError("Context feature allowlist changed")
            for features, label, identity in zip_longest(*readers):
                if None in (features, label, identity) or len({r["session_id"] for r in (features, label, identity)}) != 1:
                    raise ValueError("Context row join failed")
                key = features["session_id"]
                if key in seen or int(identity["contest_id"]) != cid:
                    raise ValueError("Duplicate/mismatched session")
                seen.add(key)
                raw[i] = [float(features[name]) if features[name] else np.nan for name in CONTEXT_COLUMNS]
                y[i] = int(label["recorded_outcome"])
                cid_vector[i] = cid
                metadata.append(identity)
                keys.append(key)
                i += 1
        finally:
            for stream in opened:
                stream.close()
    if i != n or np.isinf(raw).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid experimental table")
    return {"raw": raw, "y": y, "cid": cid_vector, "keys": keys, "metadata": metadata}


def fit_preprocessor(raw):
    stats = []
    for column in raw.T:
        known = column[~np.isnan(column)]
        stats.append({"impute": float(np.median(known)) if len(known) else 0.,
                      "mean": float(known.mean()) if len(known) else 0.,
                      "scale": float(known.std()) or 1. if len(known) else 1.})
    return {"fitted_on": "training", "rows": len(raw), "columns": list(CONTEXT_COLUMNS), "stats": stats}


def transform(raw, pre):
    if pre["fitted_on"] != "training" or pre["columns"] != list(CONTEXT_COLUMNS):
        raise ValueError("Invalid train-only normalization")
    out = np.zeros((len(raw), 2 * NFEATURES), dtype=np.float32)
    for i, stat in enumerate(pre["stats"]):
        missing = np.isnan(raw[:, i])
        out[:, i] = (np.where(missing, stat["impute"], raw[:, i]) - stat["mean"]) / stat["scale"]
        out[:, i + NFEATURES] = missing
    if not np.isfinite(out).all():
        raise ValueError("Non-finite transformed context")
    return out


def candidate_data(data, candidate):
    if candidate["history"]:
        return data
    x = data["x"].copy()
    x[:, 24:NFEATURES] = 0
    x[:, NFEATURES + 24:2 * NFEATURES] = 0
    return {**data, "x": x}


def family_score(records):
    values = [r["macro"]["validation"] for r in records]
    return (float(np.mean([v["macro_f1"] for v in values])),
            min(v["worst_contest_f1"] for v in values),
            float(np.mean([v["pooled"]["f1"] for v in values])))


def run(config_path="config/recommendation-tests-002.json"):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT / config_path).read_text())
    base = ROOT / "data" / config.get("feature_source_run", "recommendation-tests-001")
    complete = json.loads((base / "complete.json").read_text())
    models = ROOT / "data" / config["run_id"] / "models"
    report = ROOT / "reports" / config["run_id"]
    models.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    old_complete = ROOT / "data/deep-rl-2026-001/complete.json"
    manifest = {"config": config, "context_complete_sha256": file_hash(base / "complete.json"),
                "original_model_complete_sha256": file_hash(old_complete),
                "code_sha256": {name: file_hash(ROOT / name) for name in
                    ["cf_pilot/recommendation_rl.py", "cf_pilot/context.py", "tools/test_recommendations.py"]},
                "runtime": {"torch": torch.__version__, "numpy": np.__version__},
                "active_original_policy_unchanged": True}
    if config.get("reuse_unaffected_from"):
        manifest["reuse_manifest_sha256"] = file_hash(ROOT / "data" / config["reuse_unaffected_from"] / "models/manifest.json")
    frozen(models / "manifest.json", manifest)
    if (models / "complete.json").exists():
        for name, sha in json.loads((models / "complete.json").read_text())["artifacts"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError("Completed ablation artifact changed")
        print("Verified completed recommendation test cache; no refitting", flush=True)
        return
    records = {}
    candidates = {c["name"]: c for c in config["candidates"]}
    settings = config["settings"]
    old_models = ROOT / "data" / config["reuse_unaffected_from"] / "models" if config.get("reuse_unaffected_from") else None
    legacy = None
    if old_models is not None:
        old_manifest = json.loads((old_models / "manifest.json").read_text())
        archive = old_models / "source-snapshot/recommendation_rl.py"
        if file_hash(archive) != old_manifest["code_sha256"]["cf_pilot/recommendation_rl.py"]:
            raise ValueError("Archived experimental source hash differs")
        if old_manifest["config"]["settings"] != settings or old_manifest["config"]["folds"] != config["folds"]:
            raise ValueError("Cannot reuse runs with different budgets or folds")
        spec = importlib.util.spec_from_file_location("cf_pilot._archived_recommendation_001", archive)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

    def fold_data(name):
        split = config["folds"][name]
        train, val = load(split["train"], complete), load(split["validation"], complete)
        if set(train["keys"]) & set(val["keys"]):
            raise ValueError("Fold sessions overlap")
        pre = fit_preprocessor(train["raw"])
        pre["training_contests"] = split["train"]
        pre["feature_hashes"] = {str(cid): complete["contests"][str(cid)]["sha256"] for cid in split["train"]}
        frozen(models / (name + "-preprocessor.json"), pre)
        for data in (train, val):
            data["x"] = transform(data.pop("raw"), pre)
        return train, val, pre

    def fit(fold, candidate, seed, train, val):
        name = f"{fold}_{candidate['name']}_{seed}"
        metadata_path = models / (name + ".json")
        old_path = old_models / (name + ".json") if old_models is not None else None
        unaffected = not candidate["history"] or candidate["full_visible"]
        if not metadata_path.exists() and unaffected and old_path is not None and old_path.exists():
            saved = json.loads(old_path.read_text())
            if saved["candidate"] != candidate or saved["fold"] != fold or saved["seed"] != seed:
                raise ValueError("Reuse candidate metadata mismatch")
            values = candidate_data(val, candidate)
            sample = values["x"][:64]
            masks = [7] if candidate["full_visible"] else range(8)
            for mask in masks:
                np.testing.assert_array_equal(legacy.observations(sample, [mask]*len(sample)),
                                               observations(sample, [mask]*len(sample)))
            for which in ("macro", "pooled"):
                path = old_models / (name + "-" + which + ".pt")
                if file_hash(path) != saved[which]["sha256"]:
                    raise ValueError("Old control weights changed")
                weights = torch.load(path, map_location="cpu", weights_only=True)
                new_net, old_net = Network(settings["width"]), legacy.Network(settings["width"])
                new_net.load_state_dict(weights); old_net.load_state_dict(weights)
                pred, count = infer(new_net, values["x"], candidate)
                old_pred, old_count = legacy.infer(old_net, values["x"], candidate)
                np.testing.assert_array_equal(pred, old_pred)
                np.testing.assert_array_equal(count, old_count)
                if measure(values, pred, count) != saved[which]["validation"]:
                    raise ValueError("Reused control metrics do not reproduce")
                shutil.copyfile(path, models / path.name)
            saved["reused_from"] = {"metadata_path": str(old_path), "metadata_sha256": file_hash(old_path),
                "reason": "Zero history slots or full-visible mask make observations identical under the grouping correction",
                "all_validation_predictions_and_metrics_reproduced": True}
            write_json(metadata_path, saved)
            print("Reused verified unaffected control " + name, flush=True)
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            for which in ("macro", "pooled"):
                if file_hash(models / (name + "-" + which + ".pt")) != saved[which]["sha256"]:
                    raise ValueError("Cached recommendation weights changed")
        else:
            begun = time.monotonic()
            fitted = learn(candidate_data(train, candidate), candidate_data(val, candidate), candidate, settings, seed)
            for which in ("macro", "pooled"):
                torch.save(fitted[which].pop("weights"), models / (name + "-" + which + ".pt"))
                fitted[which]["sha256"] = file_hash(models / (name + "-" + which + ".pt"))
            saved = {**fitted, "fold": fold, "candidate": candidate, "seed": seed,
                     "elapsed_seconds": round(time.monotonic() - begun, 3)}
            write_json(metadata_path, saved)
        records[name] = saved

    # Fixed grid, two chronological folds, seed42. No diagnostic data loaded here.
    for fold in config["folds"]:
        train, val, _ = fold_data(fold)
        for candidate in candidates.values():
            print(f"Training development fold {fold}: {candidate['name']}", flush=True)
            fit(fold, candidate, settings["search_seed"], train, val)
        del train, val
    search = {name: family_score([v for v in records.values() if v["candidate"]["name"] == name])
              for name in candidates}
    winner = max(search, key=lambda name: search[name])
    frozen(models / "search-selection.json", {"winner": winner, "scores": {k: list(v) for k, v in search.items()},
           "diagnostic_opened": False})
    for fold in config["folds"]:
        train, val, _ = fold_data(fold)
        for name in sorted({"baseline", winner}):
            for seed in settings["repeat_seeds"]:
                print(f"Repeat development fold {fold}: {name}, seed {seed}", flush=True)
                fit(fold, candidates[name], seed, train, val)
        del train, val
    repeats = {name: family_score([v for v in records.values() if v["candidate"]["name"] == name])
               for name in sorted({"baseline", winner})}
    selected = max(repeats, key=lambda name: repeats[name])
    frozen(models / "selection.json", {"selected_candidate": selected, "search_winner": winner,
           "repeat_scores": {k: list(v) for k, v in repeats.items()}, "representative_seed": 42,
           "diagnostic_is_previously_used": True, "active_model_changed": False})
    # Previously used contest is a secondary diagnostic, never a selection input.
    diagnostic = load(config["diagnostic_contests"], complete)
    later_pre = json.loads((models / "later-preprocessor.json").read_text())
    diagnostic["x"] = transform(diagnostic.pop("raw"), later_pre)
    results = {}
    for name, meta in records.items():
        if meta["fold"] != "later":
            continue
        network = Network(settings["width"])
        network.load_state_dict(torch.load(models / (name + "-macro.pt"), map_location="cpu", weights_only=True))
        pred, count = infer(network, candidate_data(diagnostic, meta["candidate"])["x"], meta["candidate"])
        metric = measure(diagnostic, pred, count)
        for kind in sorted({row["participation_stratum"] for row in diagnostic["metadata"]}):
            mask = np.array([row["participation_stratum"] == kind for row in diagnostic["metadata"]])
            metric["participation:" + kind] = measure({"y": diagnostic["y"][mask], "cid": diagnostic["cid"][mask]}, pred[mask], count[mask])
        results[name] = metric
        if name == f"later_{selected}_42":
            with (report / "diagnostic-predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["session_id", "handle", "contest_id", "participation_stratum",
                                                            "recommendation", "recorded_outcome", "additional_inspections"])
                writer.writeheader()
                for identity, p, y, number in zip(diagnostic["metadata"], pred, diagnostic["y"], count):
                    writer.writerow({**identity, "recommendation": "REVIEW" if p else "LOW_PRIORITY",
                                     "recorded_outcome": int(y), "additional_inspections": int(number)})
    old = json.loads(old_complete.read_text())
    unchanged = all(file_hash(ROOT / name) == sha for name, sha in old["artifacts"].items())
    if not unchanged:
        raise ValueError("Original model artifacts changed")
    output = {"run_id": config["run_id"], "complete": True, "search": search,
              "search_winner": winner, "selected_candidate": selected, "repeat_scores": repeats,
              "trials": records, "diagnostic": results, "diagnostic_is_untouched": False,
              "selection_precedes_diagnostic_evaluation": True, "original_artifacts_unchanged": unchanged,
              "network_requests": 0, "new_misconduct_labels": 0,
              "interpretation": "Exploratory cached-data ablation; no independent new generalization test. Full-visible variants receive all evidence initially and have no optional inspection steps."}
    write_json(report / "results.json", output)
    files = list(models.glob("*.json")) + list(models.glob("*.pt")) + list(report.glob("*.json")) + list(report.glob("*.csv"))
    write_json(models / "complete.json", {"artifacts": {str(p.relative_to(ROOT)): file_hash(p) for p in files}})
    print("All recommendation tests completed. Selected exploratory family: " + selected, flush=True)
    print(json.dumps({"search": search, "repeat": repeats, "models": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/recommendation-tests-002.json")
    args = parser.parse_args()
    run(args.config)
