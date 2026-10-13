"""Render actual saved experiment results without training or selecting any models."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json


def plot_results(report):
    result = json.loads((report / "results.json").read_text(encoding="utf-8"))
    curves = json.loads((report / "learning_curves.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    images = []

    def save(fig, name):
        fig.savefig(report / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(report / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
        images.append(name)

    colors = {"q_learning": "#1565c0", "sarsa": "#c26513"}
    titles = {"q_learning": "Q-learning", "sarsa": "SARSA"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    for algorithm in colors:
        runs = [curves[f"{algorithm}_{s}"] for s in result["config"]["seeds"]]
        x = np.array([p["episodes"] for p in runs[0]]) / 1000
        for ax, field, label in zip(axes, ("training_reward_chunk", "validation"),
                                   ("Training reward per 50,000 episodes", "Greedy policy reward on validation")):
            data = np.array([[p[field] if field != "validation" else p[field]["mean_reward"] for p in run] for run in runs])
            mean, std = data.mean(axis=0), data.std(axis=0, ddof=1)
            ax.plot(x, mean, color=colors[algorithm], marker="o", label=titles[algorithm])
            ax.fill_between(x, mean - std, mean + std, color=colors[algorithm], alpha=.15)
            ax.set(title=label, xlabel="Training episodes (thousands)", ylabel="Mean reward")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Actual learning curves: mean and standard deviation across five seeds", fontsize=14)
    save(fig, "learning-curves")

    names = ["Always low\npriority", "Timing\nrule", "Logistic\n24 features", "Logistic\n6 summaries", "Q-learning\n5 seeds", "SARSA\n5 seeds"]
    data = [result["test"][n]["all"] for n in ("always_low_priority", "timing_rule", "logistic_full24", "logistic_matched6")]
    fig, ax = plt.subplots(figsize=(11.5, 5.2), layout="constrained")
    x = np.arange(len(names))
    for offset, field, color in ((-.24, "precision", "#1565c0"), (0, "recall", "#c26513"), (.24, "f1", "#31714b")):
        values = [d[field] for d in data] + [result["rl_seed_aggregate"][a][field]["mean"] for a in ("q_learning", "sarsa")]
        errors = [0.] * 4 + [result["rl_seed_aggregate"][a][field]["std"] for a in ("q_learning", "sarsa")]
        ax.bar(x + offset, np.array(values) * 100, .22, color=color, label=field.capitalize(),
               yerr=np.array(errors) * 100, capsize=3)
    ax.set(xticks=x, xticklabels=names, ylabel="Percent", title="Held-out recorded-outcome prediction; RL error bars show seed variation")
    ax.set_ylim(0, max(30., ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper left")
    save(fig, "test-performance")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    metrics = ["average_inspections", "mean_reward"]
    for ax, field, title in zip(axes, metrics, ("Additional evidence summaries inspected", "Held-out reward under the declared cost model")):
        values = [d[field] for d in data] + [result["rl_seed_aggregate"][a][field]["mean"] for a in ("q_learning", "sarsa")]
        errors = [0.] * 4 + [result["rl_seed_aggregate"][a][field]["std"] for a in ("q_learning", "sarsa")]
        ax.bar(x, values, color=["#88949e"] * 4 + [colors["q_learning"], colors["sarsa"]], yerr=errors, capsize=3)
        ax.set(xticks=x, xticklabels=names, title=title)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Inspection costs are simulated; they do not measure internet requests", fontsize=13)
    save(fig, "inspection-cost-and-reward")

    selected = result["selected_rl"]
    c = result["test"][selected]["all"]["confusion"]
    counts = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    normalized = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{counts[i,j]:,}\n{normalized[i,j]:.1%} of row", ha="center", va="center",
                    fontsize=14, color="white" if normalized[i,j] > .55 else "black")
    ax.set(xticks=[0, 1], xticklabels=["Low priority", "Review"],
           yticks=[0, 1], yticklabels=["No-skipped\ncomparison", "All-skipped\nmultiple problems"],
           xlabel="Policy recommendation", ylabel="Recorded outcome",
           title="Selected by validation: " + selected.replace("_", " "))
    ax.spines[:].set_visible(False)
    save(fig, "selected-policy-confusion")
    write_json(report / "plot-manifest.json", {"results_sha256": file_hash(report / "results.json"),
        "plot_code_sha256": file_hash(Path(__file__)), "images": {n + ext: file_hash(report / (n + ext))
        for n in images for ext in (".png", ".svg")}})


if __name__ == "__main__":
    config = json.loads((ROOT / "config/training-2026.json").read_text(encoding="utf-8"))
    plot_results(ROOT / "reports" / config["run_id"])
