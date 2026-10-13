# Current checkpoint

## Completed experiments

The original RL-only run `deep-rl-2026-001` remains the presentation model. Validation selected `dueling_double64_p050_42` at update 12,000. Original test contest 2274 contains 3,371 eligible sessions and 110 proxy positives. Precision is 61.68%, recall 60.00%, F1 60.83%; confusion counts are TN 3,220, FP 41, FN 44, TP 66. The small OUT_OF_COMPETITION group has only 9.09% recall.

Q-learning, SARSA and plain DQN comparisons, saved weights, action traces and plots are complete. These are recorded-outcome results, not verified cheating or AI-use results. The test is now used.

The corrected `recommendation-tests-002` study compared 18 valid RL trials across two chronological development folds. Mean development-contest F1 increased from 40.24% to 41.36% with history/all-visible, but the previously used contest diagnostic means were essentially tied, 61.9% versus 62.0%. The original policy was not replaced. An initial visibility issue was corrected and affected trials refitted; invalid trials remain excluded.

The latest pre-cleanup full suite passed 131 tests. Full experiment history, collection attempts, corrections and prior checks are preserved in [docs/archive/HISTORICAL_STATUS.md](docs/archive/HISTORICAL_STATUS.md).

## Repository cleanup and formal report

The README now follows the supplied LAB CA rubric and report format, including latent state versus observation, actions, rewards, policy, discounting, termination, architecture, implementation, parameters, training, demonstrations, metrics, graphs, limitations and references. Older guides are archived; three supporting reports are grouped under `docs/`. Aggregate report assets are published from saved results only.

The cleanup preserves original model and result bytes, source code, configurations, raw archives and unknown labels. No new model fitting, collection, active-model switch, commit or push is part of this cleanup. Post-cleanup verification is recorded below when completed.

## Next

Use the saved original model for the graded demonstration. Prepare presentation material from the README and actual results. Further enhancements need a new declared experiment and independent test after assessment.


## Post-cleanup verification completed

The existing suite passed 131 tests in 64.728 seconds. Restored-model demos reproduced true-positive, false-positive and missed-positive cases with the same action traces and rewards as the report. All local Markdown file links resolve, the formal README contains no em dashes, and Git whitespace checks passed. Three aggregate figures were visually inspected. The aggregate summary reproduces the original test confusion counts and 4,543 inspections, giving total return -21.215 and mean return -0.006293 under the declared reward. Cleanup integrity checks confirmed frozen source/config/test and original model/result bytes unchanged. Models and account-level predictions remain Git-ignored. No new production training, submission collection, commit or push occurred.


## Local review portal completed

Added a framework-free review UI under ui/, a localhost server in cf_pilot/portal.py and launcher tools/serve_portal.py. The existing frozen contest 2274 supplies 3,371 benchmark-eligible sessions and 16,693 contest-window submissions. The directory does not invent official standings. Source data are read through a verified read-only SQLite connection; model inputs come from the checksummed feature allowlist.

Run review performs actual restored dueling Double DQN inference once per server lifetime, reproducing the frozen metrics and 107 referrals. Repeated requests reuse the in-memory result. The UI supports flagged/low-priority queues, flagged-first sorting, handle search, pagination, real submission timelines, source links and policy action traces. Recorded verdicts are visible to reviewers but do not enter the model. No training, collection, source execution or label changes occurred.

Full suite: 135 tests passed in 17.867 seconds. New checks cover feature-only inference, idempotence, failed-result rejection, filtering, pagination, allowlisted static serving and local-origin/Host restrictions. Temporary headless Edge checks passed the real review button, 107-session queue, search, empty results, pagination, information dialog and mobile overflow. Desktop and mobile screenshots were visually inspected; scrollable tables and inspector were refined. The in-app CUA browser entry failed to initialize its assets, so a temporary headless Edge browser provided visual and interaction verification and was closed afterwards.

All original frozen model/report artifacts and the fresh source database still match their saved hashes. The demo is a previously evaluated snapshot, not a new held-out result. Documentation and README include the launch command and artifact prerequisites. No commit or push was made. The localhost server is restarted after checks to leave Run review available for the user's demo.
