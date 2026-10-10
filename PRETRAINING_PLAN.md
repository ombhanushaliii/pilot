# Next milestone: reviewable evidence and tested features

## Decision

Do not begin a supervised misconduct classifier or RL referral policy yet. We have submission sequences but no independently established session outcomes. A policy rewarded for high suspicion scores would learn the chosen suspicion rule; it would not establish whether misconduct was detected. Random labels or automatically treating unflagged sessions as verified fair are not acceptable.

The unit is one completed participant-contest session. The full target contest is visible at review time. This remains post-contest analysis; no live monitoring is introduced.

## 1. Build features and an evidence register in parallel

Implement a deterministic feature builder with a versioned allowlist and documented denominators. Initial features: total eligible attempts, attempted/accepted problems, same-problem retry intervals, verdict-transition counts, first-submission offset, spacing between accepted submissions, language changes and problem difficulty. Define intervals using submission timestamps, not assumed coding or judge-feedback time. Sort by timestamp with submission ID as a tie-breaker.

Compare target values with each participant's eligible earlier sessions. Use robust summaries such as median and interquartile range, with explicit missing-history flags and minimum-history rules. Retain contest duration and document elapsed-time normalization. Normalizing by duration does not make different contests equally difficult. Difficulty tags and ratings are retrospective metadata whose availability must be documented. Historical source records must predate the target; no later participation, rating or test-set statistics may alter earlier features.

Exclude identifiers as predictors, college/country, sanctions, missing-standings flags, disqualification/rating-adjustment proxies, label evidence and reviewer notes. Keep administrative verdicts such as SKIPPED out of the first feature set. Determine which ordinary verdict snapshots remain valid before including verdict-based features. Code-change features require separate verified source access and are not supported now.

Create an evidence register keyed by run ID, contest ID and handle. Record submission references, source URL, evidence type, collection/review date, reviewer decision, confidence, disagreements, any reversal, and related-case/group ID. A public allegation is a lead; a disputed decision is not silently converted into a positive. A substantiated platform/organiser finding or independently verified admission must identify the specific session. Copying evidence does not establish AI use.

Label categories: substantiated_misconduct, substantiated_ai_misuse, presumed_negative and unknown. Eligibility/excluded-participation status is separate. An ordinary reviewed session can remain presumed_negative or unknown; lack of a sanction does not establish permission/fairness. Preserve existing manual worksheets and validate evidence on import before updating database labels.

## 2. Audit the first review batch

Start with approximately 50-100 sessions to test evidence availability and the review process. Include a random component, feature-based candidates and unusual sessions with plausible explanations. Preserve sampling reasons; do not infer prevalence from this deliberately mixed batch. Seek independent positives outside the 1,000-person development sample if needed, including among the remaining target population. Existing raw historical archives can support expansion without repeating downloads.

Candidate selection may use timing/performance features. Positive labels must have evidence independent of those same feature rules. Reviewers should first assess available independent evidence without the candidate score, and record a second review where practical. Keep unresolved disagreements unknown. Record reviewed cases that cannot be resolved; do not force a balanced labelled dataset.

The batch is a feasibility audit, not a sufficient training benchmark. Count usable outcomes in each class, evidence quality and reviewer agreement. If credible positives cannot be established, the defensible output is exploratory anomaly ranking with no claim of verified cheating detection or detection accuracy.

## 3. Freeze evaluation before tuning

Repeat collection for multiple target contests with each target's own earlier panel. Round 1124 alone supports engineering development and preliminary user-held-out experiments, but cannot establish generalisation to later contests. Select future benchmark target IDs/time ranges before inspecting their model scores.

Split complete participant-contest sessions. Main evaluation uses later contests; a separate evaluation holds out participants. Keep related copying incidents/groups in one partition. Document legitimate earlier-history context for held-out sessions; target labels and future data cannot enter historical features. Declare the deployment/test sampling population and how presumed negatives are treated.

Fit scaling, imputation, feature selection and any learned reference distributions on training data only. Choose models, thresholds, reward tradeoffs and retrieval budgets on validation data. Freeze test manifests before tuning; never use a test score to improve a model and continue calling it held-out. Report small class denominators and uncertainty instead of relying on a single accuracy percentage.

## 4. Train baselines before comparing RL

Once features, labels and partitions pass audit, train simple referral rules, logistic regression and a tree-based classifier. Compare target-only inputs against full earlier-history inputs. An optional unsupervised anomaly model can help exploration before labels, but its anomaly score is neither an established cheating label nor a calibrated cheating probability. Unknown cases support demonstrations/referral rates, not correctness metrics.

RL still needs a meaningful post-contest decision: acquire earlier evidence under a measured cost/budget, then refer or finish without referral. During training, replay cached evidence and expose it only after the corresponding acquisition action. Use reviewed training outcomes for terminal feedback, not the agent's own suspicion score. Do not invent costs merely to make RL appear useful; current collection is bulk/shared, so cache and amortization must be explicit. Compare with acquire-all and simple retrieval policies, including warm-cache conditions where acquisition may be cheap.

Implement Q-learning, SARSA and DQN under matched observations, action masks and budgets after the environment is specified and tested. RL adds value only if it improves the declared error/cost tradeoff against those baselines. Repeat five seeds and report variation. Policy values are not cheating probabilities.

## Training gate

- Feature definitions, exclusions and historical cutoffs tested; adding future records leaves earlier features unchanged.
- Independent label provenance checked; presumed negatives distinguished from verified outcomes; unknown/excluded sessions do not become supervised targets.
- Enough reviewed outcomes in both classes to support declared training and held-out metrics; no magic fixed case count substitutes for this check.
- Frozen temporal and participant evaluation manifests, related-case grouping and training-only preprocessing verified.
- Reward, action availability, acquisition costs and terminal decisions specified independently of candidate-ranking heuristics.
- Shuffled-label and label-noise checks planned; no test-set tuning.

## Current status

The first feature and review pipeline is now implemented and verified; see FEATURES_AND_REVIEW.md and STATUS.md. timing_panel_v1 contains 1,000 development rows, while verdict statistics remain separate diagnostics. A 100-session random review register, submission context and validated label import are available. All real labels remain unknown and training has not started. The review-feasibility audit now determines whether reliable outcomes can support detection training, followed by additional target contests and frozen evaluation partitions. Later supervised/RL stages here remain specifications.

## Primary technical references

- Scikit-learn leakage and preprocessing guidance: https://scikit-learn.org/stable/common_pitfalls.html
- Scikit-learn outlier/novelty detection guidance: https://scikit-learn.org/stable/modules/outlier_detection.html
