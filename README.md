# Codeforces Post-Contest Review Pilot

An RL course project that analyses completed Codeforces contests and prioritizes suspicious submission patterns for human review. The planned comparison is Q-learning, SARSA and DQN against rules and supervised baselines.

The current prototype collects data, builds behavioural features and supports evidence review. Model training has not started, and all misconduct labels remain unknown.

## How we collect data

**Submission metadata comes from the official public API.** No API key is needed for public collection. Requests are sequential, at least 2.1 seconds apart, with caching, bounded retries, checksums and interruption/resume support. Failed or incomplete collection stays visible.

The active workflow starts with a completed standard Div. 2 contest and downloads its submissions and standings. It then collects the 20 preceding comparable contests within 180 days, finishing strictly before the target starts. Practice, virtual, team and out-of-competition records are retained separately from eligible individual contest sessions.

Participants are grouped by their participation in that earlier panel: regular (5+ contests), limited (1–4) and none observed (0). A deterministic 1,000-person development sample uses 70/20/10 group weights. The complete target population remains available for coverage checks. College and country are excluded from this strategy and model observations.

**Source code is collected separately for specific evidence cases.** We currently use pages opened normally in an authenticated browser: a bookmarklet exports the loaded source and submission details, or a manually saved HTML page is imported offline. Identity checks and hashes accompany every archived source. Failed attempts are relevant, so collection is not restricted to accepted submissions.

Playwright source collection was attempted, but automated browser startup failed in the agent environment and manual verification looped at Cloudflare in the separate browser profile. Unattended source scraping has not been validated. The existing Helium session can view sources, and the first manually saved page has been imported successfully. No challenge bypass is implemented.

## Progress

| Area | Current status |
| --- | --- |
| Contest collection | Round 1124, contest 2269, plus 20 earlier contests |
| Population and sample | 10,344 eligible target participants; 1,000 sampled |
| Sampled submission records | 4,286 target + 30,232 earlier = 34,518 |
| Features and review | Timing/history features, predictor allowlist and validated evidence register |
| Source feasibility | One saved HTML source imported; exporter corrected for plain-text submission IDs |
| Training and evaluation | Pending reviewed labels, more target contests and frozen data splits |
| Tests | 67 automated tests passing |

## Run locally

Python 3.11+ is sufficient for API collection, feature generation and offline source imports. Run these commands from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m cf_pilot collect --config config/contest.json
python -m cf_pilot check --run-id contest-002
python -m cf_pilot features --run-id contest-002
python -m cf_pilot review-export --run-id contest-002 --limit 100
```

Repeat `collect` to resume or reuse cached responses. Each run freezes its configuration; use a new run ID for a different collection. Data, reports, saved pages and browser profiles stay local and are excluded from Git.

Build the browser exporter with:

```powershell
python -m tools.build_source_exporter --output work/source-exporter.html
```

See [source collection](SOURCE_COLLECTION.md) for browser export and saved-HTML import instructions, [feature and review definitions](FEATURES_AND_REVIEW.md) for evidence handling, and [the project plan](PROJECT_PLAN.md) for the RL design.

## Working rules

- Preserve raw responses, provenance and checksums. Report missing history explicitly.
- Keep earlier context strictly before the target contest and fit preprocessing on training data only.
- Review evidence for the specific participant-contest session. Unflagged accounts are not verified fair examples.
- Do not turn unusual timing, `SKIPPED`, rating changes, source style or blog allegations directly into misconduct labels. Copying evidence alone does not establish AI use.
- Keep label-defining evidence and administrative outcomes out of behavioural predictors.
- Keep credentials, browser profiles, downloaded pages and collected datasets out of Git. Source collection must use access available normally.

Next: verify evidence cases, define benchmark labels, freeze temporal and participant splits, then compare baselines and RL policies under the same review costs.
