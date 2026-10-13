"""Verified numeric loader and metrics with no supervised-model dependency."""
import csv
from itertools import zip_longest
import numpy as np
from .benchmark import FEATURE_COLUMNS, TABLE_COLUMNS, file_hash


def load_data(folder, split, summary, pre):
    info = summary["partitions"][split]
    count = info["rows"]
    raw = np.empty((count, 24), dtype=np.float64)
    y = np.empty(count, dtype=np.int8)
    keys, metadata, opened = [], [], []
    try:
        readers = []
        for name in ("features", "targets", "metadata"):
            path = folder / split / (name + ".csv")
            if file_hash(path) != info["artifacts"][path.name]:
                raise ValueError("Frozen RL input changed")
            stream = path.open(encoding="utf-8", newline="")
            opened.append(stream)
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(TABLE_COLUMNS[name]):
                raise ValueError("RL input schema violates allowlist")
            readers.append(reader)
        seen, n = set(), 0
        for feature, target, identity in zip_longest(*readers):
            if None in (feature, target, identity) or n >= count:
                raise ValueError("RL partition length mismatch")
            key = feature["session_id"]
            if key in seen or key != target["session_id"] or key != identity["session_id"]:
                raise ValueError("RL session join mismatch")
            seen.add(key)
            raw[n] = [float(feature[name]) if feature[name] else np.nan for name in FEATURE_COLUMNS]
            y[n] = int(target["recorded_outcome"])
            keys.append(key)
            metadata.append(identity)
            n += 1
        if n != count or np.isinf(raw).any() or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid RL feature or target value")
    finally:
        for stream in opened:
            stream.close()
    if pre["fitted_on"] != "train" or pre["columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Preprocessing must be training-only")
    observed = ~np.isnan(raw)
    x = np.zeros((count, 48), dtype=np.float32)
    for i, name in enumerate(FEATURE_COLUMNS):
        stat = pre["stats"][name]
        x[:, i] = (np.where(observed[:, i], raw[:, i], stat["impute"]) - stat["mean"]) / stat["scale"]
        x[:, i + 24] = ~observed[:, i]
    if not np.isfinite(x).all():
        raise ValueError("Invalid RL normalized values")
    return {"x": x, "y": y, "keys": keys, "metadata": metadata}


def binary_metrics(y, prediction, inspections):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Invalid RL metrics")
    tp = int(((y == 1) & (prediction == 1)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    p, r = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    return {"sessions": len(y), "positives": tp + fn, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.,
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (r + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            "referral_rate": (tp + fp) / len(y), "average_inspections": float(np.mean(inspections))}
