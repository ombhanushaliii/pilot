"""Shuffled-reward diagnostic using the same RL learner; no classifier is fitted."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import write_json
from cf_pilot.deep_rl import learn
from cf_pilot.rl_data import binary_metrics, load_data


if __name__ == "__main__":
    config = json.loads((ROOT / "config/deep-rl.json").read_text())
    folder = ROOT / "data" / config["benchmark_id"]
    summary = json.loads((folder / "summary.json").read_text())
    pre = json.loads((folder / "preprocessor.json").read_text())
    selection = json.loads((ROOT / "data" / config["run_id"] / "selection.json").read_text())
    train, val = [load_data(folder, split, summary, pre) for split in ("train", "validation")]
    train["y"] = np.random.default_rng(31415).permutation(train["y"])
    settings = dict(config["settings"], updates=6000)
    result = learn(train, val, selection["selected_candidate"], settings, 31415, binary_metrics)
    result.pop("weights")
    write_json(ROOT / "reports" / config["run_id"] / "rl-negative-control.json", {
        "kind": "RL training outcomes shuffled, validation outcomes genuine", "seed": 31415,
        "updates": 6000, "validation": result["validation"], "history": result["history"],
        "fresh_test_not_used": True, "main_policy_changed": False,
        "interpretation": "Shorter diagnostic run, not a proof of label validity or a matched-budget algorithm comparison"})
    print(json.dumps({"RL_shuffled_reward_control": result["validation"]}, indent=2))
