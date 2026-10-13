# Tested recommendation changes

The tests found a **small development improvement, without a convincing improvement on the previously used contest**. Across six chronological development contests and three seeds, mean contest F1 rose from **40.2% to 41.4%**, a gain of **1.12 percentage points**, for historical context with all evidence visible.

On the previously used contest diagnostic, the three-seed mean F1 was **61.9% for the matched baseline and 62.0% for the history configuration**. That is essentially tied. The fixed representative seed42 was worse with history: **57.3% versus 62.6%**. We did not select a different seed or variant after seeing diagnostic results. These scores concern the recorded all-skipped outcome, not verified cheating or AI use.

The practical findings are:

- Historical context is promising on the later development split, but the earlier split is essentially tied after repeats.
- Free inspection and making all current evidence visible alone did not improve the seed42 average across both development splits.
- Keeping the best checkpoint remains useful. The matched later baseline has 35.7% mean-contest F1 at its selected checkpoint versus 32.2% at its final update.
- Selection across contests exposes failures hidden by a pooled score; it does not itself prove better performance on a new contest.
- The selected experimental seed42 still has only 9.1% recall in the small out-of-competition diagnostic group.

**Keep the original graded-presentation model for now.** The experimental weights and results are saved separately. A new, untouched and sufficiently settled test is needed before claiming better generalization or replacing the active policy. No new submissions were collected and no active-model switch was made.

Compared **18 valid RL trials**, using 20,736,000 transitions in total. Seven unaffected controls were reused after exact prediction/metric reproduction; 11 corrected-run trials were newly fitted. A first experimental run was interrupted because a derived timing feature was exposed before the pacing inspection. Its source and weights were preserved, the grouping was corrected, and the affected trials were excluded/refitted. **131 tests passed** after that fix.

The detailed methods, comparisons and limitations follow.

# Recommendation tests, 13 October 2026

Tested the proposed changes using cached Codeforces snapshots only. The original committed policy, models and results are unchanged. No new submissions were collected.

This is an exploratory development experiment. Every cached contest has already contributed to earlier analysis or training. Contest 2274 is a previously used diagnostic, not a new independent test.

## Fixed design

Five variants of dueling Double DQN: normal inspection, zero-cost inspection, all evidence visible, historical context with inspection, and historical context with all evidence visible. All share two 64-unit layers, five action outputs and the same 78-dimensional observation structure. Baseline variants zero the history slots and missing flags. There are 9,606 parameters per network, with 24 current features and 13 history/context slots. Past-only history is initially available from the cache; current gap/retry deviations are revealed with pacing/trajectory respectively. There are at most three optional current-feature inspections.

Earlier development fold: 21 training contests, validation 2257/2258/2263. Later fold: 24 training contests, validation 2264/2267/2269. Whole contests are chronological; normalization fits each fold's training features only. Histories use the last 20 earlier contests from every eligible raw session, including single-problem and ambiguous-outcome sessions. They never use verdicts, sanctions, geography or future contests. Later validation sessions may use earlier validation-contest behaviour as historical context, but no target labels.

Each run has 18,000 updates, 64 environments and 1,152,000 transitions. Episode counts differ because all-visible reference policies terminate after one action. They are contextual-bandit Q learners trained from terminal rewards, not supervised classifiers. All-visible variants receive all three optional evidence groups initially; zero additional inspection actions do not mean zero evidence was needed.

## Seed-42 development comparison

| Variant | Earlier mean contest F1 | Later mean contest F1 | Mean across both folds |
| --- | ---: | ---: | ---: |
| baseline | 44.9% | 35.7% | 40.3% |
| free_inspection | 43.5% | 35.9% | 39.7% |
| all_visible | 44.0% | 36.3% | 40.1% |
| history | 44.4% | 38.3% | 41.4% |
| history_all_visible | 44.9% | 38.3% | 41.6% |

Search winner: **history_all_visible**. Repeated that family and the baseline with seeds 736 and 2026 on both folds. The selected exploratory family after comparing repeat means is **history_all_visible**. The representative seed is fixed at 42; diagnostic outcomes never select a seed or family.

## Repeated development results

| Family | Mean contest F1 across folds/seeds | Between-seed standard deviation |
| --- | ---: | ---: |
| baseline | 40.2% | 0.08 percentage points |
| history_all_visible | 41.4% | 0.23 percentage points |

Variation across three seeds is not a cross-contest confidence interval. These development contests also selected checkpoints and families, so this is not unbiased test performance.

## Checkpoint selection control

The same learning run also saves its best pooled-F1 checkpoint. Compare it with the best mean-contest-F1 checkpoint and the final update. This changes selection, without fitting an extra network.

| Later-fold seed42 variant | Mean contest F1: selected | Mean contest F1: pooled-selected | Mean contest F1: final | Selected update |
| --- | ---: | ---: | ---: | ---: |
| baseline | 35.7% | 35.7% | 32.2% | 12,000 |
| free_inspection | 35.9% | 35.9% | 35.9% | 18,000 |
| all_visible | 36.3% | 36.3% | 36.0% | 12,000 |
| history | 38.3% | 38.3% | 37.8% | 10,000 |
| history_all_visible | 38.3% | 38.3% | 36.9% | 12,000 |

## Previously used contest diagnostic

Selection was frozen before this evaluation, but contest 2274 was already seen in prior analysis. Scores are descriptive. They do not establish new generalization or cheating-detection accuracy.

| Later-fold seed42 variant | Precision | Recall | F1 | Additional inspections |
| --- | ---: | ---: | ---: | ---: |
| baseline | 79.2% | 51.8% | 62.6% | 1.57 |
| free_inspection | 66.2% | 40.9% | 50.6% | 2.97 |
| all_visible | 75.9% | 60.0% | 67.0% | 0.00 |
| history | 65.5% | 50.0% | 56.7% | 0.93 |
| history_all_visible | 59.2% | 55.5% | 57.3% | 0.00 |

Diagnostic three-seed mean for baseline: precision 75.2%, recall 53.0%, F1 61.9%.

Diagnostic three-seed mean for history_all_visible: precision 70.0%, recall 56.1%, F1 62.0%.

Selected experimental seed42 out-of-competition diagnostic recall: 9.1%. CONTESTANT recall: 67.0%.

## Verification and limits

Compared 18 fitted RL trials and 20,736,000 valid-run environment transitions; 7 verified unaffected trials were reused from the interrupted first run. Repeated transitions are not extra people or unique data. The full automated suite passed 131 tests, including future/same-contest rejection, outcome-independent single-problem history, explicit absent history, hidden-feature masks, derived-feature visibility, matched zero placeholders, control equivalence, train-only normalization, contest selection and actual reward-based learning.

Original source databases and frozen model artifacts are unchanged. Input joins, hashes and train/validation separation are checked. Models save both macro-selected and pooled-selected checkpoints. An initial experimental run was interrupted after finding a current-gap deviation exposed before pacing inspection. Its code and weights are preserved separately. Corrected run002 reuses only unaffected controls after reproducing every validation prediction/count and both checkpoint metrics. The history-partial trials are refitted with corrected masks. This was an acquisition-visibility issue, not exposure of verdicts or targets. The primary label remains all attempts SKIPPED across multiple problems versus no SKIPPED across multiple problems; unknown misconduct labels are untouched. Recent diagnostic verdicts can change.

Historical context changes information available initially; the all-visible reference changes episode length and information exposure. These are transparent ablations, not proofs of a single causal improvement. A genuinely untouched and sufficiently settled test is still needed for a new generalization claim. No active-model switch, commit or push was made.

Feature cache: data/recommendation-tests-001. Corrected models: data/recommendation-tests-002/models. Results: reports/recommendation-tests-002. Reproduce from existing archives using tools/prepare_recommendation_tests.py, tools/test_recommendations.py and tools/report_recommendation_tests.py. Completed reruns verify cached files rather than retraining.
