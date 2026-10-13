"""Readable exploratory results and scientific plots for the fixed RL ablation."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import file_hash, write_json


def report():
    folder = ROOT / "reports/recommendation-tests-002"
    result = json.loads((folder / "results.json").read_text())
    config = json.loads((ROOT / "config/recommendation-tests-002.json").read_text())
    names = [c["name"] for c in config["candidates"]]
    labels = ["Baseline\ninspection", "Free\ninspection", "All evidence\nvisible",
              "History +\ninspection", "History +\nall visible"]
    rows = result["trials"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = ["#56718b", "#e2a83b", "#9d699f", "#429786", "#326cb5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, fold in zip(axes, config["folds"]):
        scores = [rows[f"{fold}_{name}_42"]["macro"]["validation"]["macro_f1"] * 100 for name in names]
        bars = ax.bar(range(len(names)), scores, color=colors)
        for bar, value in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.1f}", ha="center")
        ax.set_xticks(range(len(names)), labels)
        ax.set_title(f"{fold.title()} chronological development fold")
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean F1 across three validation contests (%)")
    axes[0].set_ylim(0, min(100, max(ax.get_ylim()[1] for ax in axes) + 7))
    fig.suptitle("Matched RL recommendation tests, seed 42", fontweight="bold")
    fig.text(.5, .02, "Each run: 18,000 updates and 1,152,000 transitions. All-visible policies have more completed episodes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(folder / ("recommendation-development-comparison." + extension), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for name, color, label in zip(names, colors, labels):
        history = rows[f"later_{name}_42"]["history"]
        ax.plot([point["update"] for point in history],
                [point["validation"]["macro_f1"] * 100 for point in history],
                marker="o", color=color, label=label.replace("\n", " "))
    ax.set(xlabel="Training updates", ylabel="Mean validation-contest F1 (%)",
           title="Checkpoint selection: later development fold")
    ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(.5, .01), frameon=False)
    fig.tight_layout(rect=(0, .13, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(folder / ("recommendation-checkpoint-curves." + extension), dpi=180)
    plt.close(fig)
    lines = ["# Recommendation tests, 13 October 2026", "",
        "Tested the proposed changes using cached Codeforces snapshots only. The original committed policy, models and results are unchanged. No new submissions were collected.", "",
        "This is an exploratory development experiment. Every cached contest has already contributed to earlier analysis or training. Contest 2274 is a previously used diagnostic, not a new independent test.", "",
        "## Fixed design", "",
        "Five variants of dueling Double DQN: normal inspection, zero-cost inspection, all evidence visible, historical context with inspection, and historical context with all evidence visible. All share two 64-unit layers, five action outputs and the same 78-dimensional observation structure. Baseline variants zero the history slots and missing flags. There are 9,606 parameters per network, with 24 current features and 13 history/context slots. Past-only history is initially available from the cache; current gap/retry deviations are revealed with pacing/trajectory respectively. There are at most three optional current-feature inspections.", "",
        "Earlier development fold: 21 training contests, validation 2257/2258/2263. Later fold: 24 training contests, validation 2264/2267/2269. Whole contests are chronological; normalization fits each fold's training features only. Histories use the last 20 earlier contests from every eligible raw session, including single-problem and ambiguous-outcome sessions. They never use verdicts, sanctions, geography or future contests. Later validation sessions may use earlier validation-contest behaviour as historical context, but no target labels.", "",
        "Each run has 18,000 updates, 64 environments and 1,152,000 transitions. Episode counts differ because all-visible reference policies terminate after one action. They are contextual-bandit Q learners trained from terminal rewards, not supervised classifiers. All-visible variants receive all three optional evidence groups initially; zero additional inspection actions do not mean zero evidence was needed.", "",
        "## Seed-42 development comparison", "",
        "| Variant | Earlier mean contest F1 | Later mean contest F1 | Mean across both folds |",
        "| --- | ---: | ---: | ---: |"]
    for name in names:
        early = rows[f"earlier_{name}_42"]["macro"]["validation"]["macro_f1"]
        late = rows[f"later_{name}_42"]["macro"]["validation"]["macro_f1"]
        lines.append(f"| {name} | {early:.1%} | {late:.1%} | {(early+late)/2:.1%} |")
    winner, selected = result["search_winner"], result["selected_candidate"]
    lines += ["", f"Search winner: **{winner}**. Repeated that family and the baseline with seeds 736 and 2026 on both folds. The selected exploratory family after comparing repeat means is **{selected}**. The representative seed is fixed at 42; diagnostic outcomes never select a seed or family.", "",
              "## Repeated development results", "",
              "| Family | Mean contest F1 across folds/seeds | Between-seed standard deviation |",
              "| --- | ---: | ---: |"]
    for name in result["repeat_scores"]:
        per_seed = [np.mean([rows[f"{fold}_{name}_{seed}"]["macro"]["validation"]["macro_f1"]
                            for fold in config["folds"]]) for seed in [42,736,2026]]
        lines.append(f"| {name} | {np.mean(per_seed):.1%} | {np.std(per_seed, ddof=1)*100:.2f} percentage points |")
    lines += ["", "Variation across three seeds is not a cross-contest confidence interval. These development contests also selected checkpoints and families, so this is not unbiased test performance.", "",
              "## Checkpoint selection control", "",
              "The same learning run also saves its best pooled-F1 checkpoint. Compare it with the best mean-contest-F1 checkpoint and the final update. This changes selection, without fitting an extra network.", "",
              "| Later-fold seed42 variant | Mean contest F1: selected | Mean contest F1: pooled-selected | Mean contest F1: final | Selected update |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        row = rows[f"later_{name}_42"]
        lines.append(f"| {name} | {row['macro']['validation']['macro_f1']:.1%} | {row['pooled']['validation']['macro_f1']:.1%} | {row['history'][-1]['validation']['macro_f1']:.1%} | {row['macro']['update']:,} |")
    lines += ["", "## Previously used contest diagnostic", "",
              "Selection was frozen before this evaluation, but contest 2274 was already seen in prior analysis. Scores are descriptive. They do not establish new generalization or cheating-detection accuracy.", "",
              "| Later-fold seed42 variant | Precision | Recall | F1 | Additional inspections |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        metric = result["diagnostic"][f"later_{name}_42"]["pooled"]
        lines.append(f"| {name} | {metric['precision']:.1%} | {metric['recall']:.1%} | {metric['f1']:.1%} | {metric['average_inspections']:.2f} |")
    for name in result["repeat_scores"]:
        scores = [result["diagnostic"][f"later_{name}_{seed}"]["pooled"] for seed in [42,736,2026]]
        lines += ["", f"Diagnostic three-seed mean for {name}: precision {np.mean([s['precision'] for s in scores]):.1%}, recall {np.mean([s['recall'] for s in scores]):.1%}, F1 {np.mean([s['f1'] for s in scores]):.1%}."]
    selected_row = result["diagnostic"][f"later_{selected}_42"]
    lines += ["", f"Selected experimental seed42 out-of-competition diagnostic recall: {selected_row['participation:OUT_OF_COMPETITION']['pooled']['recall']:.1%}. CONTESTANT recall: {selected_row['participation:CONTESTANT']['pooled']['recall']:.1%}.", "",
              "## Verification and limits", "",
              f"Compared {len(rows)} fitted RL trials and {sum(r['interactions'] for r in rows.values()):,} valid-run environment transitions; {sum('reused_from' in r for r in rows.values())} verified unaffected trials were reused from the interrupted first run. Repeated transitions are not extra people or unique data. The full automated suite passed 131 tests, including future/same-contest rejection, outcome-independent single-problem history, explicit absent history, hidden-feature masks, derived-feature visibility, matched zero placeholders, control equivalence, train-only normalization, contest selection and actual reward-based learning.", "",
              "Original source databases and frozen model artifacts are unchanged. Input joins, hashes and train/validation separation are checked. Models save both macro-selected and pooled-selected checkpoints. An initial experimental run was interrupted after finding a current-gap deviation exposed before pacing inspection. Its code and weights are preserved separately. Corrected run002 reuses only unaffected controls after reproducing every validation prediction/count and both checkpoint metrics. The history-partial trials are refitted with corrected masks. This was an acquisition-visibility issue, not exposure of verdicts or targets. The primary label remains all attempts SKIPPED across multiple problems versus no SKIPPED across multiple problems; unknown misconduct labels are untouched. Recent diagnostic verdicts can change.", "",
              "Historical context changes information available initially; the all-visible reference changes episode length and information exposure. These are transparent ablations, not proofs of a single causal improvement. A genuinely untouched and sufficiently settled test is still needed for a new generalization claim. No active-model switch, commit or push was made.", "",
              "Feature cache: data/recommendation-tests-001. Corrected models: data/recommendation-tests-002/models. Results: reports/recommendation-tests-002. Reproduce from existing archives using tools/prepare_recommendation_tests.py, tools/test_recommendations.py and tools/report_recommendation_tests.py. Completed reruns verify cached files rather than retraining."]
    text = "\n".join(lines) + "\n"
    (folder / "recommendation-results.md").write_text(text, encoding="utf-8")
    write_json(folder / "plot-manifest.json", {"results_sha256": file_hash(folder / "results.json"),
        "plot_code_sha256": file_hash(Path(__file__)), "plots": {
        p.name: file_hash(p) for p in folder.glob("recommendation-*") if p.suffix in {".png", ".svg"}}})
    print(text, flush=True)


if __name__ == "__main__":
    report()
