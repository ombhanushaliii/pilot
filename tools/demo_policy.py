"""Replay a saved held-out example; reveal its recorded outcome after inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.rl import STATE_FEATURES, TabularAgent, rollout, terminal_reward
from cf_pilot.training import load_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["true_positive", "true_negative", "false_positive", "false_negative"], default="true_positive")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/training-2026.json").read_text(encoding="utf-8"))
    report = ROOT / "reports" / config["run_id"]
    examples = json.loads((report / "demo_cases.json").read_text(encoding="utf-8"))
    case = examples["cases"][args.case]
    benchmark = ROOT / "data" / config["benchmark_id"]
    summary = json.loads((benchmark / "summary.json").read_text(encoding="utf-8"))
    pre = json.loads((benchmark / "preprocessor.json").read_text(encoding="utf-8"))
    data = load_split(benchmark, "test", summary, pre)
    idx = int(case["example_id"].split("_")[-1])
    model = json.loads((ROOT / "data" / config["run_id"] / (examples["selected_rl"] + ".json")).read_text(encoding="utf-8"))
    decision, count, actions = rollout(data["bins"][idx], TabularAgent.restore(model["model"]))
    if actions != case["actions"]:
        raise ValueError("Saved demonstration differs from restored policy")
    print("Post-contest recorded-outcome review demo")
    print("Example:", case["example_id"], "| Policy:", examples["selected_rl"])
    print("Initial summaries:", {name: case["observed_summaries"][name] for name in STATE_FEATURES[:3]})
    for i, action in enumerate(actions, 1):
        print(f"Action {i}: {action}")
        if action.startswith("inspect_"):
            name = STATE_FEATURES[3 + ("pacing", "trajectory", "language").index(action[8:])]
            print("Revealed summary:", name, "=", case["observed_summaries"][name])
    print("Recommendation:", "REVIEW" if decision else "LOW_PRIORITY")
    print("Additional summaries inspected:", count)
    print("Simulated inspection cost:", count * config["inspection_cost"])
    print("Recorded outcome revealed after decision:", int(data["y"][idx]))
    print("Evaluation reward including simulated cost:",
          terminal_reward(decision, int(data["y"][idx]), config["rewards"]) - count * config["inspection_cost"])
    print("Outcome interpretation: all-skipped multi-problem = 1; no-skipped comparison = 0")
    print("This output does not establish cheating or AI use.")
