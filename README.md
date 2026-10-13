# Codeforces Post-Contest Review Pilot

A reinforcement learning course project that reviews completed Codeforces contests. The agent learns which behavioural summaries to inspect, then recommends **review** or **low priority** for a participant-contest session.

The active experiment uses **only RL algorithms**: Q-learning, SARSA, DQN and dueling Double DQN. Validation selected dueling Double DQN. On one fresh contest it achieved **61.7% precision, 60.0% recall and 60.8% F1** against the recorded all-skipped multi-problem outcome. This is a measurable administrative-outcome proxy, not verified cheating or AI-use detection.

## Data collection

We collect submission metadata through Codeforces' public API, which does not require an API key. Requests are sequential, three seconds apart for the new collection, with cached responses, checksums, bounded retries and resumable checkpoints. Every selected contest is collected to an empty page. Incomplete collection fails visibly.

The current training set contains **257,660 eligible sessions from 24 contests in 2025 and 2026**. Three later contests provide 26,956 validation sessions. Contest 2274 provides 3,371 fresh test sessions. Previously evaluated test contests are now explicitly development data. The fresh contest had finished more than a day earlier, so its administrative verdicts may still change.

Positive sessions have every in-window attempt marked `SKIPPED` across at least two problems. Comparison sessions attempt at least two problems and have no skipped attempt. Ambiguous patterns stay in separate tracker tables. No recorded outcome is converted into a confirmed misconduct label.

The model sees 24 timing, attempt-sequence and language features, revealed in groups. Handles, college, country, verdicts, judge results and administrative participation status are excluded from observations. Normalization is fitted on training data only.

## What was feasible

API metadata supports automated collection and training without manual downloads. Source-code collection remains optional. Playwright verification looped at Cloudflare in a separate browser profile; normal-browser exports worked for five pilot sources. We did not establish reliable unattended source scraping. The audited public source archive had insufficient recent skipped submissions for this experiment.

## Run locally

From the repository root, with Python and the dependencies in `requirements-rl.txt`:

```powershell
python -m unittest discover -s tests -v
python tools/prepare_deep_benchmark.py
python tools/train_deep_rl.py
python tools/legacy_rl_comparison.py evaluate
python tools/plot_deep_rl.py
python tools/demo_deep_rl.py --case true_positive
python tools/demo_deep_rl.py --case false_negative
```

Preparation requires the existing verified collection archives. A completed training rerun checks saved artifacts instead of retraining or retuning on the test. The demo restores actual model weights, shows the inspection actions and reveals the recorded target only after the decision. New neural training uses one CPU thread.

Models are in `data/deep-rl-2026-001`; reports, learning curves and account-session predictions are in `reports/deep-rl-2026-001`. These local artifacts are excluded from Git. **117 automated tests pass.** See [the RL experiment](DEEP_RL.md) for reproducibility and limitations.

## Scope and working rules

- Analyse completed contests and specific sessions. Do not label an account permanently.
- Preserve raw snapshots, frozen splits, provenance and unknown misconduct labels.
- Select settings on validation, then evaluate the fresh test without retuning.
- Report precision, recall, failures and seed variation. High raw accuracy is misleading here.
- Keep credentials, browser profiles, source exports, datasets and account lists out of Git.

The current result comes from one fresh contest. Recall is only 9.1% in its small out-of-competition group, so the pooled score must not stand alone. Further claims require more untouched contests and independently established evidence.

Earlier collection checkpoints and the historical supervised comparison remain preserved. They are not part of the active RL-only training. Background: [collection and benchmark definitions](BENCHMARK.md), [source feasibility](SOURCE_COLLECTION.md), [historical experiment](TRAINING.md), [project plan](PROJECT_PLAN.md).
