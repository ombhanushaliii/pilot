"""Offline training and held-out evaluation of recorded outcomes, not cheating truth."""

import csv
from itertools import zip_longest
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .benchmark import FEATURE_COLUMNS, TABLE_COLUMNS, file_hash, write_json
from .rl import STATE_FEATURES, REWARDS, TabularAgent, rollout
from .store import canonical


def load_split(folder, split, summary, pre):
    """Load identities for reporting only; the numeric matrices have a strict allowlist."""
    metadata = summary["partitions"][split]
    n = metadata["rows"]
    streams = []
    try:
        readers = []
        for name in ("features", "targets", "metadata"):
            path = folder / split / (name + ".csv")
            if file_hash(path) != metadata["artifacts"][path.name]:
                raise ValueError("Frozen benchmark artifact changed")
            stream = path.open(encoding="utf-8", newline="")
            streams.append(stream)
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(TABLE_COLUMNS[name]):
                raise ValueError("Input schema violates feature allowlist")
            readers.append(reader)
        raw, y = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float64), np.empty(n, dtype=np.int8)
        keys, meta, seen = [], [], set()
        count = 0
        for feature, label, identity in zip_longest(*readers):
            if None in (feature, label, identity) or count >= n:
                raise ValueError("Unequal or unexpected partition length")
            key = feature["session_id"]
            if key in seen or key != label["session_id"] or key != identity["session_id"]:
                raise ValueError("Duplicate or mismatched session identity")
            seen.add(key)
            raw[count] = [float(feature[name]) if feature[name] else np.nan for name in FEATURE_COLUMNS]
            target = int(label["recorded_outcome"])
            if target not in (0, 1):
                raise ValueError("Invalid recorded-outcome target")
            y[count] = target
            keys.append(key)
            meta.append(identity)
            count += 1
        if count != n or np.isinf(raw).any():
            raise ValueError("Invalid numeric feature data")
    finally:
        for stream in streams:
            stream.close()
    if pre["columns"] != list(FEATURE_COLUMNS) or pre["fitted_on"] != "train":
        raise ValueError("Preprocessing is not the frozen training-only state")
    values, missing = np.empty_like(raw), np.isnan(raw)
    for i, name in enumerate(FEATURE_COLUMNS):
        stat = pre["stats"][name]
        values[:, i] = (np.where(missing[:, i], stat["impute"], raw[:, i]) - stat["mean"]) / stat["scale"]
    x = np.concatenate((values, missing.astype(float)), axis=1).astype(np.float32)
    bins = np.empty((n, len(STATE_FEATURES)), dtype=np.int8)
    for i, name in enumerate(STATE_FEATURES):
        column = raw[:, FEATURE_COLUMNS.index(name)]
        bins[:, i] = np.where(np.isnan(column), 4,
                             np.searchsorted(pre["stats"][name]["quartiles"], column, side="left"))
    if not np.isfinite(x).all():
        raise ValueError("Invalid standardized features")
    return {"raw": raw, "x": x, "bins": [tuple(row) for row in bins.tolist()],
            "y": y, "keys": keys, "metadata": meta}


def scores(model, x):
    columns = model["input_indices"]
    z = x[:, columns] @ np.asarray(model["coefficients"], dtype=float) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -500., 500.)))


def metrics(y, prediction, inspections, config):
    y, pred = np.asarray(y), np.asarray(prediction)
    counts = np.broadcast_to(np.asarray(inspections, dtype=float), y.shape)
    if y.shape != pred.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Invalid metric inputs")
    tp, fp = int(np.sum((pred == 1) & (y == 1))), int(np.sum((pred == 1) & (y == 0)))
    tn, fn = int(np.sum((pred == 0) & (y == 0))), int(np.sum((pred == 0) & (y == 1)))
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    r = config["rewards"]
    total = tp * r["true_positive"] + fp * r["false_positive"] + tn * r["true_negative"] + fn * r["false_negative"]
    return {"sessions": len(y), "positives": tp + fn, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "accuracy": (tp + tn) / len(y),
            "balanced_accuracy": (recall + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            "referral_rate": (tp + fp) / len(y), "average_inspections": float(np.mean(counts)),
            "average_steps": float(np.mean(counts)) + 1,
            "average_inspection_cost": float(np.mean(counts)) * config["inspection_cost"],
            "mean_reward": total / len(y) - float(np.mean(counts)) * config["inspection_cost"]}


def select_threshold(probabilities, y, config):
    """Optimize validation reward over tied-score groups, including no referrals."""
    p, y = np.asarray(probabilities), np.asarray(y)
    if p.shape != y.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid validation scores")
    order = np.argsort(-p, kind="stable")
    ordered, target = p[order], y[order]
    r = config["rewards"]
    delta = np.where(target == 1, r["true_positive"] - r["false_negative"],
                     r["false_positive"] - r["true_negative"])
    cumulative = np.cumsum(delta)
    ends = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True])
    # Lower referral counts win exact reward ties.
    best, threshold = 0., float(np.nextafter(ordered[0], np.inf))
    for i in ends:
        if cumulative[i] > best:
            best, threshold = float(cumulative[i]), float(ordered[i])
    return threshold


def rollout_batch(data, agent):
    cache, predictions, inspections = {}, [], []
    for row in data["bins"]:
        if row not in cache:
            cache[row] = rollout(row, agent)
        decision, acquired, actions = cache[row]
        predictions.append(decision)
        inspections.append(acquired)
    return np.asarray(predictions, dtype=np.int8), np.asarray(inspections, dtype=np.int8)


def train_logistic(train, val, name, indices, config, destination):
    best, candidates = None, []
    for c in config["logistic_c"]:
        fit = LogisticRegression(C=c, solver="lbfgs", max_iter=500, tol=1e-4, random_state=config["seeds"][0])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            fit.fit(train["x"][:, indices], train["y"])
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise ValueError("Logistic regression did not converge; do not publish an unfinished fit")
        model = {"name": name, "algorithm": "logistic_regression", "c": c, "input_indices": indices,
                 "coefficients": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0]),
                 "iterations": fit.n_iter_.tolist(), "inspection_count": 3}
        prob = scores(model, val["x"])
        model["threshold"] = select_threshold(prob, val["y"], config)
        measurement = metrics(val["y"], prob >= model["threshold"], 3, config)
        candidates.append({"c": c, "validation": measurement, "threshold": model["threshold"]})
        if best is None or measurement["mean_reward"] > best["validation"]["mean_reward"]:
            best = {"model": model, "validation": measurement}
    best["candidates"] = candidates
    write_json(destination, best)
    return best


def rule_predictions(model, data):
    gap = data["raw"][:, FEATURE_COLUMNS.index("gap_median_fraction")]
    retry = data["raw"][:, FEATURE_COLUMNS.index("retry_fraction")]
    return ((gap <= model["gap_threshold"]) & (retry >= model["retry_threshold"])).astype(np.int8)


def train_rule(val, pre, config, destination):
    best, candidates = None, []
    for gap in sorted(set(pre["stats"]["gap_median_fraction"]["quartiles"])):
        for retry in sorted(set(pre["stats"]["retry_fraction"]["quartiles"])):
            model = {"algorithm": "timing_rule", "gap_threshold": gap,
                     "retry_threshold": retry, "inspection_count": 2}
            measurement = metrics(val["y"], rule_predictions(model, val), 2, config)
            candidates.append({"gap": gap, "retry": retry, "validation": measurement})
            if best is None or measurement["mean_reward"] > best["validation"]["mean_reward"]:
                best = {"model": model, "validation": measurement}
    best["candidates"] = candidates
    write_json(destination, best)
    return best


def train_agent(train, val, algorithm, seed, config, destination, progress=print):
    agent = TabularAgent(algorithm, alpha=config["alpha"], gamma=config["gamma"], seed=seed,
                         budget=config["budget"])
    rng = np.random.default_rng(seed)
    # Separate row RNG from action RNG gives both algorithms identical session samples.
    indices = rng.integers(0, len(train["y"]), size=config["episodes"])
    history, best, running = [], None, 0.
    start = time.monotonic()
    for episode, idx in enumerate(indices, 1):
        fraction = min(1., (episode - 1) / config["epsilon_decay_episodes"])
        epsilon = config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])
        running += agent.episode(train["bins"][int(idx)], int(train["y"][idx]), epsilon,
                                 cost=config["inspection_cost"], rewards=config["rewards"])
        if episode % config["checkpoint_interval"] == 0 or episode == config["episodes"]:
            pred, acquired = rollout_batch(val, agent)
            measurement = metrics(val["y"], pred, acquired, config)
            point = {"episodes": episode, "epsilon": epsilon,
                     "training_reward_chunk": running / config["checkpoint_interval"], "validation": measurement,
                     "visited_states": len(agent.q)}
            history.append(point)
            running = 0.
            if best is None or measurement["mean_reward"] > best["validation"]["mean_reward"]:
                best = {"model": agent.export(), "selected_episode": episode, "validation": measurement}
            progress(f"{algorithm} seed {seed}: {episode:,} episodes, validation reward {measurement['mean_reward']:.3f}, recall {measurement['recall']:.3f}", flush=True)
    best.update(seed=seed, history=history, elapsed_seconds=round(time.monotonic() - start, 3))
    write_json(destination, best)
    return best


def sliced_metrics(data, prediction, inspections, config, training_handles, validation_handles):
    result = {"all": metrics(data["y"], prediction, inspections, config)}
    masks = {}
    for stratum in sorted({m["participation_stratum"] for m in data["metadata"]}):
        masks["participation:" + stratum] = np.asarray([m["participation_stratum"] == stratum for m in data["metadata"]])
    for contest in sorted({m["contest_id"] for m in data["metadata"]}):
        masks["contest:" + contest] = np.asarray([m["contest_id"] == contest for m in data["metadata"]])
    masks["unseen_in_train"] = np.asarray([m["handle"] not in training_handles for m in data["metadata"]])
    previously_seen = training_handles | validation_handles
    masks["unseen_in_train_and_validation"] = np.asarray([m["handle"] not in previously_seen for m in data["metadata"]])
    counts = np.broadcast_to(np.asarray(inspections), data["y"].shape)
    for name, mask in masks.items():
        if mask.any():
            result[name] = metrics(data["y"][mask], prediction[mask], counts[mask], config)
    return result


def train_run(root, config, progress=print):
    root = Path(root)
    benchmark = root / "data" / config["benchmark_id"]
    summary = json.loads((benchmark / "summary.json").read_text(encoding="utf-8"))
    pre = json.loads((benchmark / "preprocessor.json").read_text(encoding="utf-8"))
    if not summary["complete"] or file_hash(benchmark / "preprocessor.json") != summary["preprocessor_sha256"]:
        raise ValueError("Benchmark incomplete or preprocessing modified")
    if pre["training_features_sha256"] != summary["partitions"]["train"]["artifacts"]["features.csv"]:
        raise ValueError("Preprocessing training provenance mismatch")
    model_dir = root / "data" / config["run_id"]
    report = root / "reports" / config["run_id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "benchmark_summary_sha256": file_hash(benchmark / "summary.json"),
                "preprocessor_sha256": file_hash(benchmark / "preprocessor.json"),
                "code_sha256": {name: file_hash(root / name) for name in ("cf_pilot/rl.py", "cf_pilot/training.py")},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                "state_features": list(STATE_FEATURES), "tabular_hidden_value": -1, "tabular_missing_value": 4,
                "selection_metric": "validation mean reward", "cost_is_simulated": True,
                "target_is_misconduct_truth": False, "training_sampling": "natural frequency, uniform sessions with replacement"}
    frozen = model_dir / "manifest.json"
    if frozen.exists() and canonical(json.loads(frozen.read_text(encoding="utf-8"))) != canonical(manifest):
        raise ValueError("Frozen training run differs; use a new run_id")
    write_json(frozen, manifest)
    completion = model_dir / "complete.json"
    if completion.exists():
        previous = json.loads(completion.read_text(encoding="utf-8"))
        for relative, sha in previous["artifacts"].items():
            if file_hash(root / relative) != sha:
                raise ValueError("Completed model or result artifact changed")
        progress("Completed training run verified; no retraining or test retuning", flush=True)
        return json.loads((report / "results.json").read_text(encoding="utf-8"))
    write_json(model_dir / "status.json", {"stage": "training"})
    progress("Loading verified training and validation data; test remains unopened", flush=True)
    train, val = [load_split(benchmark, p, summary, pre) for p in ("train", "validation")]
    if set(train["keys"]) & set(val["keys"]):
        raise ValueError("Training/validation session overlap")
    learners, validation = {}, {}
    matched = [FEATURE_COLUMNS.index(name) for name in STATE_FEATURES]
    index_sets = {"logistic_full24": list(range(2 * len(FEATURE_COLUMNS))),
                  "logistic_matched6": matched + [i + len(FEATURE_COLUMNS) for i in matched]}
    for name, indices in index_sets.items():
        path = model_dir / (name + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
        else:
            progress("Training " + name, flush=True)
            saved = train_logistic(train, val, name, indices, config, path)
        learners[name], validation[name] = saved, saved["validation"]
    rule_path = model_dir / "timing_rule.json"
    saved_rule = json.loads(rule_path.read_text(encoding="utf-8")) if rule_path.exists() else train_rule(val, pre, config, rule_path)
    learners["timing_rule"], validation["timing_rule"] = saved_rule, saved_rule["validation"]
    for algorithm in ("q_learning", "sarsa"):
        for seed in config["seeds"]:
            name = f"{algorithm}_{seed}"
            path = model_dir / (name + ".json")
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else train_agent(
                train, val, algorithm, seed, config, path, progress)
            learners[name], validation[name] = saved, saved["validation"]
    # Freeze deployment selection before accessing final test data.
    rl_names = [n for n in learners if learners[n]["model"].get("algorithm") in {"q_learning", "sarsa"}]
    selected = max(rl_names, key=lambda n: validation[n]["mean_reward"])
    selection = {"selected_rl": selected, "metric": "validation mean reward", "validation": validation,
                 "model_sha256": {name: file_hash(model_dir / (name + ".json")) for name in learners}}
    selection_path = model_dir / "selection.json"
    if selection_path.exists() and canonical(json.loads(selection_path.read_text(encoding="utf-8"))) != canonical(selection):
        raise ValueError("Frozen model selection differs")
    write_json(selection_path, selection)
    write_json(model_dir / "status.json", {"stage": "final_test", "selection_sha256": file_hash(selection_path)})
    progress("Selection frozen. Loading held-out test contests for final evaluation", flush=True)
    test = load_split(benchmark, "test", summary, pre)
    if set(test["keys"]) & (set(train["keys"]) | set(val["keys"])):
        raise ValueError("Final-test session overlap")
    training_handles = {m["handle"] for m in train["metadata"]}
    validation_handles = {m["handle"] for m in val["metadata"]}
    results, selected_predictions, selected_inspections = {}, None, None
    results["always_low_priority"] = sliced_metrics(test, np.zeros(len(test["y"]), dtype=np.int8), 0, config,
                                                  training_handles, validation_handles)
    for name, saved in learners.items():
        model = saved["model"]
        if model["algorithm"] == "logistic_regression":
            prediction = (scores(model, test["x"]) >= model["threshold"]).astype(np.int8)
            inspections = np.full(len(test["y"]), 3, dtype=np.int8)
        elif model["algorithm"] == "timing_rule":
            prediction = rule_predictions(model, test)
            inspections = np.full(len(test["y"]), 2, dtype=np.int8)
        else:
            prediction, inspections = rollout_batch(test, TabularAgent.restore(model))
        results[name] = sliced_metrics(test, prediction, inspections, config, training_handles, validation_handles)
        if name == selected:
            selected_predictions, selected_inspections = prediction, inspections
    aggregate = {}
    for algorithm in ("q_learning", "sarsa"):
        members = [results[f"{algorithm}_{s}"]["all"] for s in config["seeds"]]
        aggregate[algorithm] = {field: {"mean": float(np.mean([m[field] for m in members])),
                                      "std": float(np.std([m[field] for m in members], ddof=1))}
                                for field in ("precision", "recall", "f1", "mean_reward", "average_inspections", "referral_rate")}
    selected_agent = TabularAgent.restore(learners[selected]["model"])
    csv_path = report / "test_predictions.csv"
    examples = {}
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session_id", "handle", "contest_id", "recommendation",
            "recorded_outcome", "participation_stratum", "inspections", "simulated_cost", "actions"])
        writer.writeheader()
        for i, identity in enumerate(test["metadata"]):
            decision, acquired, actions = rollout(test["bins"][i], selected_agent)
            if decision != selected_predictions[i] or acquired != selected_inspections[i]:
                raise ValueError("Restored policy prediction mismatch")
            row = {**identity, "recommendation": "REVIEW" if decision else "LOW_PRIORITY",
                   "recorded_outcome": int(test["y"][i]), "inspections": acquired,
                   "simulated_cost": acquired * config["inspection_cost"], "actions": " > ".join(actions)}
            writer.writerow(row)
            category = ("true_positive" if decision else "false_negative") if test["y"][i] else (
                "false_positive" if decision else "true_negative")
            if category not in examples:
                examples[category] = {"example_id": "heldout_" + str(i), "recorded_outcome": int(test["y"][i]),
                                     "recommendation": row["recommendation"], "actions": actions,
                                     "simulated_cost": row["simulated_cost"],
                                     "observed_summaries": {STATE_FEATURES[j]: float(test["raw"][i, FEATURE_COLUMNS.index(STATE_FEATURES[j])])
                                         if np.isfinite(test["raw"][i, FEATURE_COLUMNS.index(STATE_FEATURES[j])]) else None
                                         for j in range(6) if j < 3 or ("inspect_" + ("pacing", "trajectory", "language")[j - 3]) in actions}}
    output = {"run_id": config["run_id"], "benchmark_id": config["benchmark_id"], "complete": True,
              "config": config, "validation": validation, "selected_rl": selected, "test": results,
              "rl_seed_aggregate": aggregate, "selection_sha256": file_hash(selection_path),
              "test_opened_after_selection": True, "network_requests": 0, "misconduct_labels_assigned": 0,
              "state_abstraction": "6 summaries with train-fitted quartile bins; matched logistic uses same 6 numeric summaries",
              "interpretation": "Recorded-outcome prediction with simulated evidence-inspection costs, not verified cheating accuracy"}
    write_json(report / "results.json", output)
    write_json(report / "demo_cases.json", {"selected_rl": selected, "cases": examples})
    write_json(report / "learning_curves.json", {n: v["history"] for n, v in learners.items() if "history" in v})
    write_json(model_dir / "status.json", {"stage": "complete"})
    artifacts = {str(p.relative_to(root)): file_hash(p) for p in list(model_dir.glob("*.json")) + list(report.glob("*"))
                 if p.is_file() and p.name != "complete.json"}
    write_json(completion, {"artifacts": artifacts})
    progress("Training and final evaluation complete: " + selected, flush=True)
    return output
