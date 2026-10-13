# Local review portal

Pilot's demo UI uses the **real cached contest 2274** and the original saved dueling Double DQN. No synthetic predictions, source downloads, extra collection or training are required.

## Start

From the repository root, in the existing Python environment with `requirements-rl.txt` installed:

```powershell
python tools/serve_portal.py
```

Open **http://127.0.0.1:8765** in your browser. Use `--port 8766` if the default port is occupied. Press Ctrl+C in the terminal to stop the server.

The portal needs the local frozen `data/deep-benchmark-2026-001`, `data/deep-rl-2026-001`, `data/rl-expansion-2026-002/pilot.sqlite` and `reports/deep-rl-2026-001` artifacts. A fresh Git clone excludes these files and cannot run the model until the verified artifacts are available. Missing or changed artifacts cause a visible startup error.

## Demonstrate

1. Browse the participant directory and select a session. Its real submission sequence appears in the inspector.
2. Click **Run review**. The server restores the original policy and performs actual inference on 3,371 sessions using behavioural inputs only.
3. The interface opens **Flagged**, containing the 107 review recommendations, and shows the policy's inspection trace for the selected session.
4. Search a handle, switch to **All** or **Low priority**, and compare submission trails. A low-priority recommendation does not establish fair participation.

One inference pass is cached in memory for each server lifetime. Repeat requests reuse it; reloading the browser does not retrain or recompute. Restart the server to reset the demonstration. No model or dataset is modified, and no account list is written to Git.

## Interpretation

This is a benchmark-eligible session directory, **not an official ranklist**. The snapshot does not contain standings. Directory positions are alphabetical before review; flags are listed first afterwards. Filtering does not alter model decisions.

The contest has already been evaluated, so this is a saved-model demonstration rather than a new generalization test. Recorded submission verdicts are visible to the reviewer but excluded from the model's observations. Review priority does not establish cheating or AI use. The known model output is 107 referrals, including 66 proxy positives and 41 comparison sessions, with 44 missed proxy positives.

The UI uses local HTML, CSS and JavaScript. The server uses the Python standard library plus the existing RL dependencies. There is no frontend framework, external font download, browser scraping or build step. The service binds only to localhost.
