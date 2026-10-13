"""Publish aggregate report evidence from saved results without fitting or collecting."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build():
    report = ROOT / "reports/deep-rl-2026-001"
    destination = ROOT / "docs/assets"
    destination.mkdir(parents=True, exist_ok=True)
    results = json.loads((report / "results.json").read_text())
    curves = json.loads((report / "learning-curves.json").read_text())
    plots = json.loads((report / "plot-manifest.json").read_text())
    if digest(report / "results.json") != plots["results_sha256"]:
        raise ValueError("Results differ from saved plot provenance")
    published = {}
    for source, name in {
        "rl-only-fresh-comparison.png": "rl-comparison.png",
        "rl-only-selected-confusion.png": "confusion-matrix.png",
    }.items():
        if digest(report / source) != plots["plots"][source]:
            raise ValueError("Saved source figure changed")
        shutil.copyfile(report / source, destination / name)
        published[name] = digest(destination / name)

    points = curves[results["selected"]]
    episodes = [p["finished_episodes"] / 1000 for p in points]
    selected = results["selection"]["selected_update"]
    selected_point = next(p for p in points if p["updates"] == selected)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    measurements = [
        ([p["training_reward_per_interaction"] for p in points], "Cumulative training reward per interaction", "Mean reward"),
        ([100 * p["validation"]["f1"] for p in points], "Greedy validation F1", "F1 (%)"),
        ([1 + p["validation"]["average_inspections"] for p in points], "Greedy validation episode length", "Mean actions"),
    ]
    for ax, (values, title, ylabel) in zip(axes, measurements):
        ax.plot(episodes, values, marker="o", color="#1565c0", linewidth=2)
        ax.axvline(selected_point["finished_episodes"] / 1000, color="#a55413", linestyle="--", label="Selected at 12,000 updates")
        ax.set(title=title, xlabel="Completed training episodes (thousands)", ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[2].legend(loc="lower left", fontsize=8)
    figure.suptitle("Selected dueling Double DQN run: recorded checkpoints, seed 42", fontsize=14)
    for suffix in ("png", "svg"):
        path = destination / ("learning-progress." + suffix)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        published[path.name] = digest(path)
    plt.close(figure)

    chosen = results["fresh_test"][results["selected"]]["all"]
    with (report / "fresh-test-predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    inspections = sum(int(row["inspections"]) for row in rows)
    confusion = {name: 0 for name in ("tn", "fp", "fn", "tp")}
    for row in rows:
        target, predicted = int(row["recorded_outcome"]), int(row["recommendation"] == "REVIEW")
        key = ("tp" if predicted else "fn") if target else ("fp" if predicted else "tn")
        confusion[key] += 1
    if confusion != chosen["confusion"] or len(rows) != chosen["sessions"]:
        raise ValueError("Prediction CSV does not reproduce saved confusion counts")
    if abs(inspections / len(rows) - chosen["average_inspections"]) > 1e-12:
        raise ValueError("Inspection counts differ from saved metric")
    total_return = confusion["tp"] - confusion["fn"] - .5 * confusion["fp"] - .005 * inspections
    legacy = json.loads((report / "legacy-rl-fresh-test.json").read_text())
    summary = {
        "scope": "Recorded all-skipped multi-problem outcome; not confirmed misconduct or AI use",
        "run_id": results["run_id"], "selected": results["selected"], "selected_update": selected,
        "population": {"training_sessions": results["training_sessions"], "validation_sessions": results["validation_sessions"], "test_sessions": results["test_sessions"]},
        "selected_validation": results["validation"][results["selected"]],
        "selected_original_test": results["fresh_test"][results["selected"]],
        "repeat_seed_aggregate": results["repeat_seed_aggregate"],
        "tabular_original_test_seed_aggregate": legacy["seed_aggregate"],
        "total_selected_test_inspections": inspections,
        "selected_test_total_return": total_return,
        "selected_test_mean_return": total_return / len(rows),
        "selected_test_mean_actions": 1 + inspections / len(rows),
        "always_low_test_mean_return": -chosen["positives"] / len(rows),
        "source_sha256": {name: digest(report / name) for name in ("results.json", "learning-curves.json", "fresh-test-predictions.csv", "legacy-rl-fresh-test.json")},
        "aggregate_only": True,
    }
    path = destination / "results-summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    published[path.name] = digest(path)
    manifest = {"generated_by": "tools/build_report_assets.py", "code_sha256": digest(Path(__file__)), "published_sha256": published, "no_fitting": True, "no_collection": True}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"published": list(published), "total_return": total_return, "mean_return": total_return / len(rows)}))


if __name__ == "__main__":
    build()
