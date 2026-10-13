# RL-only experiment: deep-rl-2026-001

The active policy is dueling Double DQN, selected using validation F1. It predicts a recorded all-skipped multi-problem outcome from verdict-free behaviour and decides which cached evidence groups to inspect before ending with review or low priority. No supervised classifier is fitted or used by this training entry point. The previous logistic experiment is historical.

## Versioned data

| Partition | Contests | Eligible sessions | Positive | Comparison |
| --- | ---: | ---: | ---: | ---: |
| Training | 24 | 257,660 | 7,721 | 249,939 |
| Validation | 3 | 26,956 | 768 | 26,188 |
| Fresh test | 1 | 3,371 | 110 | 3,261 |

Training combines six preserved 2025 contests and 18 earlier 2026 contests. Validation uses 2264, 2267 and 2269, which were tested previously and are explicitly reclassified as development data. Fresh test is 2274, collected before training and selected without inspecting its verdict counts. It finished on 11 October 2026; collection was on 13 October. The original seven-day recency audit found no newer suitable mature Div. 2 contest. A separate configuration declares a one-day minimum for this fresh test. Recorded administrative outcomes may change after this snapshot.

Whole contests are chronologically separated. Returning handles are allowed in the main evaluation; 708 fresh sessions have handles unseen in both training and validation. These are observed handles, not verified identities. Current CONTESTANT and OUT_OF_COMPETITION groups are reported separately and never become features.

Positive means all recorded in-window attempts are SKIPPED across at least two problems. Comparison means at least two attempted problems and no SKIPPED. Exclude single-problem, replacement-compatible, partial-final and unresolved patterns from the primary binary task, retaining them in trackers. These rules define eligibility, not the reason for a skip. Misconduct labels remain unknown.

Verified raw API archives, separate features/targets/metadata and immutable source hashes are retained. Original databases were not modified. Full contest archives, rather than the earlier 1,000-person sample, supply this experiment. API requests are serial at three-second intervals. Source code is unused.

## Environment and agent

An episode represents offline review of one completed session. Five basic aggregates are visible initially: attempts, attempted problems, first and last submission time fractions and attempts per hour. Three inspection actions reveal pacing (seven features), trajectory (eight) or language (four). Review and low priority terminate the episode. Each inspection can occur once, with a maximum of three inspections and four actions. This is a finite sequential evidence-inspection environment; it does not control contestants or monitor a live contest.

The 24 features are normalized with training-only statistics and accompanied by 24 missing-value indicators. Hidden groups, including their missing indicators, are zeroed. Three visibility bits and a remaining-budget fraction give a 52-dimensional observation. Fixed clipping is [-8, 8]. Identifiers, final verdicts, judge fields, administrative status and profile fields do not enter inference. The target is used for rewards and class-aware replay sampling during training only.

Selected rewards: positive review +1; missed positive -1; comparison review -0.5; comparison low priority 0; each inspection -0.005. Gamma is 1 because episodes are finite and terminate. These costs are declared assumptions, not measured API or human-review costs. All target submissions are already cached.

The selected network has two shared 64-unit ReLU layers, a scalar state-value head and five action-advantage outputs, with 7,942 parameters. Dueling combines value and centered advantages. Double DQN chooses the next valid action with the online network and evaluates it with a separate target network. Terminal transitions never bootstrap; repeated inspection actions are masked in both acting and targets. Learning uses Bellman temporal-difference targets, not classification loss. Primary algorithm references: [Double DQN](https://arxiv.org/abs/1509.06461), [dueling networks](https://arxiv.org/abs/1511.06581).

## Training and selection

Frozen config: `config/deep-rl.json`. Five candidates use seed 42: plain DQN (64 units, penalty 0.25), dueling Double DQN (64 units, penalties 0.10/0.25/0.50) and dueling Double DQN (128 units, penalty 0.25). The winning configuration is repeated with seeds 736 and 2026. The primary selected seed remains 42; test results never choose a different seed.

Each model runs 18,000 updates with 64 parallel environment instances, totaling 1,152,000 transitions per model. Seven models produce 8,064,000 transitions and 4,129,683 completed episodes. These reuse the 257,660 training sessions; they are not additional people or unique records. A checkpoint is evaluated every 2,000 updates and selected by validation F1, then precision to break ties. Selected primary checkpoint is update 12,000.

AdamW learning rate 0.001, weight decay 0.0001, SmoothL1 TD loss, gradient clipping 5, soft target update tau 0.01, batch 256, warmup 500 updates. Epsilon decreases from 1 to 0.05 over 80% of updates. Replay holds up to 50,000 transitions per outcome class, sampled equally with weights based on natural training-session class frequency. Reset episodes use natural uniform session sampling. Validation and test retain their natural frequencies. Processing uses one Torch/BLAS CPU thread, PyTorch 2.6.0+cpu, NumPy 2.2.5 and Python 3.13.2.

Model weights, selection, config, code hashes and preprocessing are saved before fresh-test evaluation. Saved artifacts are checked on cache reuse. A larger network or later checkpoint was not automatically better.

## Results on the same fresh contest

| RL algorithm | Seeds reported | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Q-learning | Five-seed mean | 47.3% | 63.8% | 54.2% |
| SARSA | Five-seed mean | 47.3% | 63.1% | 53.7% |
| Plain DQN | Seed 42, validation-selected checkpoint | 50.7% | 65.5% | 57.1% |
| Selected dueling Double DQN | Seed 42, validation-selected checkpoint | 61.7% | 60.0% | 60.8% |

The selected policy refers 107 of 3,371 sessions: 66 match the positive target and 41 belong to the comparison group. It misses 44 positive sessions and assigns low priority to 3,220 comparison sessions. Mean additional inspections: 1.35. Selected validation F1 was 35.5%, substantially lower than the fresh-test score; this illustrates contest variation.

Across three dueling Double DQN seeds, fresh F1 is 59.7% with sample standard deviation 1.4 percentage points. Precision is 70.7% plus/minus 9.5 points and recall 52.7% plus/minus 7.3 points. The selected seed is not replaced by the higher-precision seed after seeing test results. Q-learning F1 standard deviation is 2.8 points; SARSA is 5.3 points. Seed variation is not a cross-contest confidence interval.

Among 708 handles unseen in training/validation, precision is 70.1%, recall 67.5% and F1 68.8%, with 80 positive outcomes. Different prevalence prevents direct generalization to other populations. CONTESTANT recall is 72.7% (64/88); OUT_OF_COMPETITION recall is only 9.1% (2/22). The latter has no comparison sessions and cannot establish precision specificity. This material weakness must appear in the presentation.

Historical tabular agents use six discretized summaries and an older, smaller training population. Neural agents use 24 continuous features and expanded training data. Their common fresh test is useful operationally, but the difference cannot be attributed solely to the algorithm. The matched neural-width-64 penalty-0.25 DQN/dueling-Double pair is available in the full results; it also changes both Double and dueling components, so it does not isolate either one.

## Verification and limits

117 automated tests pass, including hand-computed DQN/Double targets, terminal handling, hidden-feature invariance, invalid-action masks, replay weighting and actual learning on a controlled hidden-feature task. CSV confusion counts reproduce the saved metrics. All frozen artifacts and source fingerprints are checked. Four plots were visually inspected. Restored-model demos passed positive, false-positive and missed-positive cases; the missed-positive demo inspects all three groups before deciding low priority.

A training-label-shuffled RL control ran 6,000 updates without accessing the fresh test. Genuine-validation F1 was 14.4%, below the main model's 35.5%. It has a shorter budget and retains signal after checkpoint selection, so it is a diagnostic rather than proof of no leakage or a fair main-model comparison.

The best claim is best validation F1 among the tested RL settings. One recent fresh contest with 110 positives cannot establish universal superiority or cheating accuracy. SKIPPED and absence of SKIPPED do not establish AI use or innocence. Raw accuracy is misleading: an always-low policy already scores 96.7% with zero recall. Further improvement must use new versioned development data and a new untouched test; do not tune against this test repeatedly.

## Reproduce and demonstrate

```powershell
python tools/prepare_deep_benchmark.py
python tools/train_deep_rl.py
python tools/legacy_rl_comparison.py evaluate
python tools/plot_deep_rl.py
python tools/demo_deep_rl.py --case true_positive
python tools/demo_deep_rl.py --case false_positive
python tools/demo_deep_rl.py --case false_negative
```

Existing verified source archives are required for preparation. Collection snapshots, reports and models remain local and Git-ignored. Training cache reuse verifies frozen files, with no retraining or new test-based tuning. The historical tabular selection and weights are preserved; evaluation requires that saved selection. The primary output is the review recommendation plus inspected groups and action trace. Q values are action values, not calibrated probabilities of cheating.
