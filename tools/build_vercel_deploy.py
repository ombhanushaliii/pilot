"""Build the public API portal with frozen weights and training preprocessing."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
STAGE = WORK / "vercel-deploy"
sys.path.insert(0, str(ROOT))

from cf_pilot.benchmark import FEATURE_COLUMNS, FEATURE_GROUPS
from cf_pilot.deep_rl import infer
from cf_pilot.portal import Portal
from cf_pilot.vercel_portal import RemotePortal


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    if STAGE.parent.resolve() != WORK.resolve() or WORK.parent.resolve() != ROOT.resolve():
        raise RuntimeError("Refusing an unexpected staging location")
    WORK.mkdir(exist_ok=True)
    project_link = (STAGE / ".vercel" / "project.json").read_bytes() if (STAGE / ".vercel" / "project.json").exists() else None
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "api" / "_private").mkdir(parents=True)
    if project_link:
        (STAGE / ".vercel").mkdir()
        (STAGE / ".vercel" / "project.json").write_bytes(project_link)
    portal = Portal(ROOT)

    rows = []
    for row in portal.rows:
        rows.append({
            "id": row["id"], "handle": row["handle"], "attempts": row["attempts"],
            "problem_count": row["problem_count"], "last_seconds": row["last_seconds"],
            "submissions": [{key: item[key] for key in ("id", "problem", "problem_name", "language", "verdict", "seconds")}
                            for item in row["submissions"]],
        })
    snapshot = {
        "contest": portal.contest,
        "snapshot_time": portal.snapshot_time,
        "model": portal.selection["selected"],
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "rows": rows,
    }
    payload = STAGE / "api" / "_private"
    copy(portal.folder / "preprocessor.json", payload / "preprocessor.json")
    (payload / "model.json").write_text(json.dumps({"selected": portal.selection["selected"]}), encoding="utf-8")
    selected_weights = torch.load(portal.models / (portal.selection["selected"] + ".pt"), map_location="cpu", weights_only=True)
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in selected_weights.items()}
    np.savez_compressed(payload / "policy.npz", **arrays)

    expected_decisions, expected_inspections, _, expected_traces = infer(portal.network, portal.data["x"], traces=True)
    remote = RemotePortal(payload, snapshot, portal.data["x"])
    result = remote.infer()
    for index, row in enumerate(rows):
        actual = result["decisions"][row["id"]]
        expected_status = "review" if expected_decisions[index] else "low_priority"
        expected_actions = [
            ("inspect_pacing", "inspect_trajectory", "inspect_language", "low_priority", "review")[action]
            for action in expected_traces[index]
        ]
        if actual["status"] != expected_status or actual["inspections"] != int(expected_inspections[index]) or actual["actions"] != expected_actions:
            raise RuntimeError(f"NumPy deployment policy differs from the saved PyTorch policy at session {row['id']}")
    if any(key in snapshot for key in ("targets", "labels", "outcomes")) or any("verdict" in row for row in rows):
        raise RuntimeError("Deployment snapshot must not contain training targets")

    copy(ROOT / "api" / "index.py", STAGE / "api" / "index.py")
    copy(ROOT / "cf_pilot" / "vercel_portal.py", STAGE / "cf_pilot" / "vercel_portal.py")
    for name in ("live_portal", "benchmark", "features", "audit", "sessions", "store", "outcomes"):
        copy(ROOT / "cf_pilot" / (name + ".py"), STAGE / "cf_pilot" / (name + ".py"))
    copy(ROOT / "cf_pilot" / "__init__.py", STAGE / "cf_pilot" / "__init__.py")
    for source, name in (("remote.html", "index.html"), ("remote_app.js", "app.js"),
                         ("styles.css", "styles.css"), ("remote.css", "remote.css"), ("favicon.svg", "favicon.svg")):
        copy(ROOT / "ui" / source, STAGE / name)
    (STAGE / "requirements.txt").write_text("numpy==2.2.5\n", encoding="utf-8")
    (STAGE / ".python-version").write_text("3.13\n", encoding="utf-8")
    config = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "functions": {"api/index.py": {"maxDuration": 60, "includeFiles": "api/_private/**"}},
        "rewrites": [{"source": "/api/:path*", "destination": "/api?route=:path*"}],
    }
    (STAGE / "vercel.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in STAGE.rglob("*") if path.is_file())
    print(f"Built {STAGE}")
    print(f"Sessions: {len(rows):,}; submissions: {sum(row['attempts'] for row in rows):,}; policy parity: exact across all sessions/actions")
    print(f"Package files: {total / (1024 * 1024):.2f} MiB; weights and preprocessing only, no bundled contest or training data")


if __name__ == "__main__":
    main()
