> Historical record. The current scope and formal report are in [the main README](../../README.md). This document does not override the active RL-only experiment.

# Practical completion route

The detailed active plan is now [HF_BENCHMARK_PLAN.md](HF_BENCHMARK_PLAN.md). In particular, use complete API sessions to define negative outcomes and investigate ordinary resubmission-related skips. SKIPPED is not an explicit enforcement-reason label. This earlier proposal remains a record of the initial metadata-only route.

## Decision proposed after the source pilot

Stop requesting manual source downloads. The source pilot demonstrates acquisition, but verifying AI misuse from it is not a feasible prerequisite for finishing this course project.

Use an automatically observable benchmark target: whether a completed official participant-contest session has at least one submission whose archived verdict is SKIPPED. Call the target observed_skipped, never verified_cheating or verified_ai_use. The negative class means no SKIPPED observed in that snapshot, not fair participation. Keep the existing misconduct labels unknown and create a separate outcome table/manifest.

The research question becomes: can behavioural submission patterns and strictly earlier contest context prioritise sessions associated with this platform outcome, and can an RL policy match a full-history baseline using fewer evidence groups?

## Data already available

A read-only audit on 11 October 2026 found:

| Available normalized sessions | Count | At least one SKIPPED |
| --- | --- | --- |
| Full target contest 2269 | 10,344 | 272 |
| Target development sample | 1,000 | 21 |
| Earlier sessions for sampled participants | 6,622 | 106 |

These are snapshot outcomes, not misconduct findings or prevalence estimates. The earlier sample was chosen from target-contest participants and has selection limitations. All 21 contests also have complete raw response archives. Audit and build a separate benchmark from those archives rather than treating only the development sample as the full population. Counts for the full historical populations have not yet been computed.

## Automated work order

1. Stream the cached archives and count eligible official individual sessions and observed_skipped outcomes per contest. Preserve checksums and snapshot dates. No new source-page collection.
2. Freeze a benchmark configuration and population. Use the oldest five contests for historical context, the next ten for training, the next five for validation, and contest 2269 for the final temporal test. Confirm dates and counts before adopting this partition. Use an additional participant-disjoint test. Historical context must always predate its target session.
3. Build timing, attempt, retry, language-switch and earlier-history features for each benchmark session. Start with the existing timing allowlist. Exclude all final verdicts, SKIPPED counts, sanctions, missing-standings indicators, rating adjustments, identifiers, organisation, nationality and blog/source evidence. Administrative verdict diagnostics are not predictors.
4. Train rules and logistic regression or a small tree model. Handle class imbalance using training-only weights or sampling. Keep validation/test at their observed outcome proportions. Freeze thresholds using validation only.
5. Implement a short post-contest RL episode: observe complete target timing features, optionally reveal prior timing or participation summaries, then refer or finish without referral. Mask repeated/unavailable acquisitions and expose missing-history status consistently. No live monitoring.
6. Compare Q-learning and SARSA first, then a small CPU DQN. Terminal rewards use observed_skipped outcomes. The reward tradeoff for acquisition is an explicitly simulated evidence-group budget, not a claimed saving in API calls. Evaluate zero acquisition penalty as well as several nonzero penalties. All data are cached and full target data are available from episode start.
7. Report temporal and participant-disjoint results over five seeds: precision-recall, recall at a fixed review budget, precision among top-ranked sessions, observed_skipped-negative referral rate, acquisitions per episode and return. Compare target-only, acquire-all and random-acquisition policies under the same benchmark. If RL does not improve the tradeoff, report that result.
8. Deliver scripts, saved configs/models, learning curves, a ranked post-contest demonstration, README and the course report. Output referral priority, decision and factual timing/history reasons. No calibrated cheating probability.

## Tests that matter

- Reconcile session/outcome counts against raw archives and recorded eligibility.
- Verify feature output is unchanged when final verdicts are masked or altered.
- Verify future submissions cannot influence earlier features.
- Verify session/participant partitions, training-only preprocessing and action masks.
- Keep outcome labels separate from observations and real misconduct labels.
- Run a shuffled-outcome control, class-imbalance checks and five seeds.
- Do not generate synthetic data or label outlier scores as cheating to obtain favourable accuracy.

## Limits and fallback

This evaluates prediction of an archived platform verdict, not proof of AI misuse, plagiarism or fair participation. Dataset snapshots do not show exactly when the verdict changed, so the demonstration is retrospective. Avoid claims of prospective detection accuracy without appropriately timed snapshots. Verdict reversals and platform procedures can introduce noise.

If temporal/user-disjoint outcome counts are too small, add completed contests automatically through the existing API collector. If an outcome model performs poorly, retain an unsupervised anomaly-ranking demonstration and honestly report the RL benchmark result. Isolation Forest is an optional baseline, not an RL algorithm or a source of verified labels.

This route is proposed and audited at the count level. No benchmark outcomes have been materialized and no training has run yet. Manual review notes and misconduct labels remain intact.
