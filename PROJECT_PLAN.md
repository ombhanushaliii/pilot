# Current project scope

The active project is a completed RL-only prototype for **post-contest Codeforces session review**. Its formal report is [README.md](README.md); detailed model and enhancement methods are in [docs/DEEP_RL.md](docs/DEEP_RL.md) and [docs/RECOMMENDATION_TESTS.md](docs/RECOMMENDATION_TESTS.md).

## Frozen presentation checkpoint

- Dataset: `deep-benchmark-2026-001`. Training: 257,660 sessions from 24 contests; validation: 26,956 sessions from three contests; original test: 3,371 sessions from contest 2274.
- Model: `deep-rl-2026-001`, `dueling_double64_p050_42`, checkpoint 12,000. Original test precision 61.68%, recall 60.00%, F1 60.83%.
- RL comparison: Q-learning, SARSA, plain DQN and dueling Double DQN. Historical classifier work is not active.
- Target: all in-window attempts SKIPPED across at least two problems versus no skipped attempts across at least two problems. This is an observed-outcome proxy; misconduct labels remain unknown.
- Recommendation experiments are complete. Their small development gain did not establish better independent performance, so the original policy remains active.

## Remaining assessment work

Demonstrate restored-model inference, including false positives and false negatives, and use saved learning curves and comparisons in the presentation. Complete the institution's cover-sheet fields when submitting the report. No further training or data collection is planned for the presentation.

## Rules for future work

Keep verdicts, judge results, profiles, identity and geography out of observations. Fit preprocessing on training data only, require earlier context to predate the target contest, preserve frozen raw responses and models, and report exclusions and subgroup failures. Inspection costs are simulated over cached evidence.

Contest 2274 and the six enhancement-development contests have already been used. A future improvement claim requires a predeclared design and an untouched, sufficiently settled test. Never replace the selected seed using the most favourable known-test result.

The complete preceding plans are preserved in [docs/archive/HISTORICAL_PROJECT_PLAN.md](docs/archive/HISTORICAL_PROJECT_PLAN.md), including designs that were superseded. They do not override this scope.
