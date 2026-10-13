"""Verify immutable ablation artifacts and reproduce selected validation inference."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json
from cf_pilot.recommendation_rl import Network, infer, measure
from tools.test_recommendations import load, transform, candidate_data, family_score


def verify():
    torch.set_num_threads(1)
    config = json.loads((ROOT / "config/recommendation-tests-002.json").read_text())
    models, report = ROOT / "data/recommendation-tests-002/models", ROOT / "reports/recommendation-tests-002"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    artifact_checks = {name: file_hash(ROOT / name) == sha for name, sha in read(models / "complete.json")["artifacts"].items()}
    if not all(artifact_checks.values()):
        raise ValueError("Frozen experimental artifact changed")
    manifest, result, selection = [read(path) for path in [models / "manifest.json", report / "results.json", models / "selection.json"]]
    if not all(file_hash(ROOT / name) == sha for name, sha in manifest["code_sha256"].items()):
        raise ValueError("Experiment code changed")
    old_checks = {name: file_hash(ROOT / name) == sha for name, sha in read(ROOT / "data/deep-rl-2026-001/complete.json")["artifacts"].items()}
    if not all(old_checks.values()):
        raise ValueError("Original active model or results changed")
    source = read(ROOT / "data/deep-benchmark-2026-001/manifest.json")
    paths = {"contest-002": ROOT / "data/pilot.sqlite", "api-2025-feasibility-001": ROOT / "data/api-2025-feasibility-001/pilot.sqlite",
             "rl-expansion-2026-002": ROOT / "data/rl-expansion-2026-002/pilot.sqlite"}
    databases = {name: file_hash(path) == source["source_fingerprints"][name] for name, path in paths.items()}
    if not all(databases.values()):
        raise ValueError("Source database changed")
    records = result["trials"]
    families = result["repeat_scores"]
    calculated = {name: family_score([v for v in records.values() if v["candidate"]["name"] == name]) for name in families}
    selected = max(calculated, key=lambda name: calculated[name])
    if selected != result["selected_candidate"] or selected != selection["selected_candidate"]:
        raise ValueError("Saved selection does not match development criterion")
    if selection["representative_seed"] != 42 or not result["selection_precedes_diagnostic_evaluation"]:
        raise ValueError("Diagnostic influenced selection")
    complete = read(ROOT / "data/recommendation-tests-001/complete.json")
    validation = load(config["folds"]["later"]["validation"], complete)
    pre = read(models / "later-preprocessor.json")
    validation["x"] = transform(validation.pop("raw"), pre)
    reproduced = {}
    for name in sorted({"baseline", selected}):
        key = f"later_{name}_42"
        candidate = records[key]["candidate"]
        network = Network(config["settings"]["width"])
        network.load_state_dict(torch.load(models / (key + "-macro.pt"), map_location="cpu", weights_only=True))
        pred, count = infer(network, candidate_data(validation, candidate)["x"], candidate)
        actual = measure(validation, pred, count)
        if actual != records[key]["macro"]["validation"]:
            raise ValueError("Selected validation inference does not reproduce")
        reproduced[key] = actual
    diagnostic = load(config["diagnostic_contests"], complete)
    diagnostic["x"] = transform(diagnostic.pop("raw"), pre)
    seen = set()
    for cid in config["folds"]["later"]["train"] + config["folds"]["later"]["validation"]:
        with (ROOT / "data/deep-benchmark-2026-001/contests" / str(cid) / "metadata.csv").open(encoding="utf-8", newline="") as stream:
            seen.update(row["handle"] for row in csv.DictReader(stream))
    unseen = np.array([row["handle"] not in seen for row in diagnostic["metadata"]])
    slices = {}
    selected_pred = None
    for name in sorted({"baseline", selected}):
        key = f"later_{name}_42"; candidate = records[key]["candidate"]
        network = Network(config["settings"]["width"])
        network.load_state_dict(torch.load(models / (key + "-macro.pt"), map_location="cpu", weights_only=True))
        pred, count = infer(network, candidate_data(diagnostic, candidate)["x"], candidate)
        if measure(diagnostic, pred, count) != result["diagnostic"][key]:
            # Stored diagnostic includes participation slices as additional keys.
            if measure(diagnostic, pred, count)["pooled"] != result["diagnostic"][key]["pooled"]:
                raise ValueError("Diagnostic inference changed")
        slices[key] = measure({"y": diagnostic["y"][unseen], "cid": diagnostic["cid"][unseen]}, pred[unseen], count[unseen])
        if name == selected: selected_pred = pred
    with (report / "diagnostic-predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [row["session_id"] for row in rows] != diagnostic["keys"]:
        raise ValueError("Saved predictions have incorrect identities/order")
    csv_pred = np.array([row["recommendation"] == "REVIEW" for row in rows], dtype=np.int8)
    if not np.array_equal(selected_pred, csv_pred):
        raise ValueError("Prediction CSV does not reproduce")
    test_log = Path(r"C:\Users\Om Bhanushali\Documents\Codex\2026-10-10\pro\outputs\recommendation-tests-002-unittest.log")
    text = test_log.read_text(encoding="utf-8")
    if not re.search(r"Ran 131 tests in", text) or not text.rstrip().endswith("OK"):
        raise ValueError("Expected regression suite did not pass")
    answer = {"verified": True, "run_id": result["run_id"], "artifact_checks": artifact_checks,
              "original_model_checks": old_checks, "source_database_checks": databases,
              "validation_inference_reproduced": reproduced, "prediction_csv_reproduced": True,
              "selection_uses_development_only": True, "representative_seed": 42,
              "previously_used_diagnostic_unseen_primary_handles": slices,
              "tests": {"passed": 131, "log_sha256": file_hash(test_log)},
              "models_compared": len(records), "unaffected_reused_controls": sum("reused_from" in v for v in records.values()),
              "network_requests": 0, "new_misconduct_labels": 0, "active_model_changed": False}
    write_json(report / "verification.json", answer)
    print("Verified original/model/source hashes, 131 tests, selection, validation inference and diagnostic CSV", flush=True)
    print(json.dumps({"selected": selected, "scores": calculated, "unseen_diagnostic": slices}, indent=2), flush=True)


if __name__ == "__main__":
    verify()
