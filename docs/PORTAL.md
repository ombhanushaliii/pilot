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

## Remote demo on Vercel

The public portal is deployed at [pilot-review-five.vercel.app](https://pilot-review-five.vercel.app). Paste a completed Codeforces contest URL or contest ID, choose **Load contest**, then **Run review**. No portal password or Codeforces API key is required. Round 1127 uses [contest 2274](https://codeforces.com/contest/2274); [open it directly in the portal](https://pilot-review-five.vercel.app/?contest=2274).

The Vercel backend calls `contest.standings` and pages through `contest.status` sequentially, with at least 2.1 seconds between request starts on each server instance. It filters to individual official and out-of-competition submissions created during the actual contest window. Practice, virtual and team sessions are excluded. Single-problem sessions remain visible as outside model scope; sessions with at least two attempted problems are scored. Verdicts are shown in the inspector but never enter the model. Live scope includes ambiguous administrative outcomes, so the scored population can differ from the historical outcome-labelled benchmark.

Behavioural features use the unchanged training-fitted preprocessor and saved dueling Double DQN weights. No fitting occurs on the website. Summary responses omit submission detail lists; the inspector fetches a selected participant's details separately to stay within serverless response limits. A snapshot fingerprint prevents applying decisions to a changed collection. Fetch errors, incomplete collection, unfinished contests and pre-2025 contests produce visible errors; there is no bundled-contest fallback.

The backend keeps up to three fetched contests for ten minutes in a warm process. Cold or different server instances may fetch again, so this is not a global cache or a global API rate limiter. Collection is bounded to 150,000 returned submissions and a 43-second collection budget. Larger or slow contests fail rather than being silently truncated. Vercel responses are capped below its function payload limit. Administrative verdicts can change after a contest, and live results are not a new accuracy evaluation.

Rebuild with `python tools/build_vercel_deploy.py`. It checks exact decision and action-trace parity with the original PyTorch model on all 3,371 frozen test sessions, then exports only weights, training normalization settings, code and web assets. No contest snapshot, feature matrix, raw database, training partitions or targets are packaged. The build preserves the staging project's Vercel link.

```powershell
Set-Location .\work\vercel-deploy
vercel deploy --prod --yes
```

Review results remain in the open browser page. Vercel executes actual inference when **Run review** is clicked. The trained model files remain on the backend. Vercel's project-level deployment protection settings are unchanged.
