# Automated 2026 recorded outcome benchmark

This version uses all eligible individual sessions from the full 21-contest cached 2026 panel. The original 1,000-person development sample is not the sampling population. No network requests or source-code downloads are needed to build it.

Run `python tools/build_2026_benchmark.py` from the repository root. The configuration is frozen in `config/benchmark-2026.json`, and the source run is `contest-002`. The builder opens the source SQLite database read-only, verifies raw response hashes, validates complete pagination checkpoints and checks the database fingerprint again before completing. Older checkpoints without a highest-ID field derive their boundary from the verified first archived page. Interrupted builds resume from checked per-contest outputs; changed configurations, archives or completed outputs fail visibly.

The oldest 15 cached contests form training. Validation uses 2257, 2258 and 2263. Final testing uses 2264, 2267 and 2269. All split boundaries are chronological. These are the newest archived contests, not necessarily the site's latest contests today. Whole sessions stay in one partition. Returning users are allowed in the primary temporal experiment; overlap and unseen-user test counts are reported for an additional generalization assessment.

## Outcome definition

A positive recorded outcome means every eligible in-contest attempt is SKIPPED across at least two problems. Comparisons have no SKIPPED attempts and also attempt at least two problems. Single-problem, replacement-compatible, partial-final-skipped and unresolved sessions are excluded from the initial binary benchmark and retained in tracker files. Current contestant and out-of-competition strata are kept separately. No skipped outcome establishes fair participation, and an all-skipped outcome does not establish the reason for skipping or AI use. All misconduct labels remain unknown.

## Local artifacts

`data/benchmark-2026-001` contains the frozen source manifest, build status, per-contest outputs, partition tables, preprocessing and aggregate summary. `reports/benchmark-2026-001/summary.json` contains the aggregate report. Data and reports remain outside Git.

Each partition has three aligned tables:

- `features.csv`: session join key and 24 allowlisted behavioural values.
- `targets.csv`: session join key and recorded-outcome target.
- `metadata.csv`: handle, contest and current participation stratum for joins and reporting.

Per-contest `tracker.csv` also keeps excluded session patterns and unknown misconduct labels. A target column is not an input feature. The session key, handle, contest ID, participation stratum and split identity must never be fed to the model. Load only `FEATURE_COLUMNS` from `cf_pilot.benchmark` and enforce that allowlist.

## Features and preprocessing

Four groups describe basic attempt timing, pacing/bursts, problem-attempt trajectories and language changes. The function reads submission timestamps, IDs for ordering, problem indices for grouping, language, and contest start/duration. It does not use verdicts, judging fields, problem ratings, standings, personal profiles, source style or sanctions. Timestamp gaps describe submission activity, not time spent writing code. Current problem names/indices are not categorical model inputs.

`preprocessor.json` is fitted only from training features and records the training-file checksum, observed/missing counts, medians, means, scales and quartile boundaries. The transform emits 24 normalized values plus 24 missing-value indicators. Missing retry or language-transition statistics remain explicit. Constant columns use scale 1; completely unobserved columns use a declared zero fallback and retain missing masks. Validation/test are transformed using that saved training state. No labels are used in preprocessing.

## Next stage

Train supervised references and Q-learning/SARSA evidence-inspection agents on this version. Keep final test outcomes out of parameter selection. Declare simulated inspection costs, compare against simple and acquire-all baselines, report natural test class balance and strata, and describe results as recorded-outcome prediction. A policy value is not a cheating probability. DQN can be added after the first two agents and evaluation work.
