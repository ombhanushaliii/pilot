"""Fixed shuffled-label diagnostic; does not select or modify deployed models."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json
from cf_pilot.training import load_split, metrics, scores, select_threshold


if __name__ == "__main__":
    config = json.loads((ROOT / "config/training-2026.json").read_text(encoding="utf-8"))
    folder = ROOT / "data" / config["benchmark_id"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    pre = json.loads((folder / "preprocessor.json").read_text(encoding="utf-8"))
    source = ROOT / "data" / config["run_id"] / "logistic_matched6.json"
    original = json.loads(source.read_text(encoding="utf-8"))["model"]
    train, val = [load_split(folder, split, summary, pre) for split in ("train", "validation")]
    shuffled = np.random.default_rng(31415).permutation(train["y"])
    fit = LogisticRegression(C=original["c"], solver="lbfgs", max_iter=500, tol=1e-4, random_state=31415)
    fit.fit(train["x"][:, original["input_indices"]], shuffled)
    model = {"coefficients": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0]),
             "input_indices": original["input_indices"]}
    probability = scores(model, val["x"])
    threshold = select_threshold(probability, val["y"], config)
    validation = metrics(val["y"], probability >= threshold, 3, config)
    test = load_split(folder, "test", summary, pre)
    measurement = metrics(test["y"], scores(model, test["x"]) >= threshold, 3, config)
    write_json(ROOT / "reports" / config["run_id"] / "negative-control.json", {
        "kind": "matched6 logistic with training labels shuffled", "seed": 31415,
        "c": original["c"], "threshold_selected_on": "validation", "threshold": threshold,
        "validation": validation, "test": measurement, "main_models_modified": False,
        "reference_model_sha256": file_hash(source), "control_code_sha256": file_hash(Path(__file__)),
        "interpretation": "A signal diagnostic, not proof of label validity or a complete leakage audit"})
    print(json.dumps({"negative_control": measurement}, indent=2))
