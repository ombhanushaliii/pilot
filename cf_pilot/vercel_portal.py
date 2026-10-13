"""Small, stateless remote inference service for Vercel."""
import json
from pathlib import Path
import time

import numpy as np


ACTIONS = ("inspect_pacing", "inspect_trajectory", "inspect_language", "low_priority", "review")


class RemotePortal:
    def __init__(self, payload, snapshot=None, x=None):
        payload = Path(payload)
        self.snapshot = snapshot if snapshot is not None else json.loads((payload / "snapshot.json").read_text(encoding="utf-8"))
        with np.load(payload / "policy.npz", allow_pickle=False) as archive:
            self.x = np.asarray(x if x is not None else archive["x"], dtype=np.float32).reshape((-1, 48))
            self.weights = {name: archive[name].astype(np.float32, copy=False) for name in archive.files if name != "x"}
        self.rows = self.snapshot["rows"]
        if len(self.rows) != len(self.x) or self.x.shape[1] != 48:
            raise ValueError("Remote policy snapshot has an incompatible feature matrix")
        self.columns = self.snapshot["feature_columns"]
        self.groups = self.snapshot["feature_groups"]
        self.starts = {name: i for i, name in enumerate(self.columns)}
        self.visibility = np.zeros((8, 48), dtype=np.float32)
        for mask in range(8):
            visible = set(self.groups["basic"])
            for index, name in enumerate(("pacing", "trajectory", "language")):
                if mask & (1 << index):
                    visible.update(self.groups[name])
            for name in visible:
                index = self.starts[name]
                self.visibility[mask, index] = 1.
                self.visibility[mask, index + 24] = 1.
        self.bits = np.asarray([[(mask >> i) & 1 for i in range(3)] for mask in range(8)], dtype=np.float32)

    def _forward(self, x):
        w = self.weights
        hidden = np.maximum(x @ w["shared.0.weight"].T + w["shared.0.bias"], 0)
        hidden = np.maximum(hidden @ w["shared.2.weight"].T + w["shared.2.bias"], 0)
        advantage = hidden @ w["advantage.weight"].T + w["advantage.bias"]
        value = hidden @ w["value.weight"].T + w["value.bias"]
        return value + advantage - advantage.mean(axis=1, keepdims=True)

    def infer(self):
        started = time.perf_counter()
        n = len(self.rows)
        decisions = np.zeros(n, dtype=np.int8)
        inspections = np.zeros(n, dtype=np.int8)
        masks = np.zeros(n, dtype=np.int8)
        traces = [[] for _ in range(n)]
        active = np.arange(n)
        for step in range(4):
            if not len(active):
                break
            visible = np.clip(self.x[active], -8., 8.) * self.visibility[masks[active]]
            remaining = ((3 - self.bits[masks[active]].sum(axis=1)) / 3).astype(np.float32)[:, None]
            observation = np.concatenate((visible, self.bits[masks[active]], remaining), axis=1).astype(np.float32)
            q = self._forward(observation)
            invalid = self.bits[masks[active]].astype(bool)
            q[:, :3][invalid] = -np.inf
            action = q.argmax(axis=1)
            for row_index, selected in zip(active, action):
                traces[int(row_index)].append(ACTIONS[int(selected)])
            terminal = action >= 3
            ended = active[terminal]
            decisions[ended] = (action[terminal] == 4).astype(np.int8)
            inspections[ended] = step
            active = active[~terminal]
            masks[active] |= (1 << action[~terminal]).astype(np.int8)
        if len(active):
            raise ValueError("Saved policy exceeded its inspection budget")
        flagged = int(decisions.sum())
        return {"contest_id": self.snapshot["contest"]["id"], "participants": n, "flagged": flagged,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "decisions": {row["id"]: {"status": "review" if decisions[i] else "low_priority",
                    "inspections": int(inspections[i]), "actions": traces[i]} for i, row in enumerate(self.rows)}}

    def public_snapshot(self):
        rows = getattr(self, "directory", self.rows)
        return {"contest": self.snapshot["contest"], "snapshot_time": self.snapshot["snapshot_time"],
                "participants": len(rows), "eligible_participants": len(self.rows),
                "submissions": sum(row["attempts"] for row in rows),
                "snapshot_id": self.snapshot.get("snapshot_id"), "model": self.snapshot["model"],
                "rows": [{k: v for k, v in row.items() if k != "submissions"} for row in rows]}
