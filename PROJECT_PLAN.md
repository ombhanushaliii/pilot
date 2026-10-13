# Post contest submission review project

## Active RL-only implementation, 13 October 2026

The user requires reinforcement learning algorithms for the active project. Implemented and trained plain DQN and dueling Double DQN, retaining Q-learning and SARSA as RL comparisons. The selected production policy is dueling_double64_p050_42, chosen by validation F1 at update 12,000. No supervised classifier is used by the active tools/train_deep_rl.py pipeline. Historical classifier work remains archived.

Use deep-benchmark-2026-001 and config/deep-benchmark-2026.json: 24 earlier 2025/2026 contests and 257,660 training sessions, three development contests and 26,956 validation sessions, fresh contest 2274 and 3,371 test sessions. Previously tested 2264/2267/2269 are development data now. Collection 2274 uses a separately declared one-day minimum after an outcome-blind mature-contest audit found no newer suitable seven-day-old contest. Recent administrative verdicts can change.

Positive means all recorded in-window attempts skipped across at least two problems; comparison means no skipped attempts across at least two problems. Ambiguous and single-problem sessions stay in trackers. Misconduct labels remain unknown. Every source snapshot/database and older frozen run is preserved. The full archived contest population supplies training, not the earlier 1,000-person sample.

The offline finite environment starts with five basic behavioural summaries. Actions inspect pacing, trajectory or language once each, then review or low priority. Costs are simulated inspection costs over cached features, not measured API savings. Neural observations include 24 normalized features with missing and visibility masks; identity, geography, verdicts, judge results and administrative status are excluded. Training-only preprocessing, temporal contest splits and unseen-handle reporting are enforced. Read DEEP_RL.md for reward choices, Bellman targets, exploration, replay, selection, actual comparisons and limitations.

Training and fresh evaluation are complete. Seven neural models produced 8,064,000 interactions. Selected fresh precision is 61.7%, recall 60.0%, F1 60.8%; three-seed F1 mean is 59.7%. OUT_OF_COMPETITION recall is only 9.1% on 22 positives. This is a one-contest recorded-outcome result, not validated cheating detection. The test is now used and cannot be retuned as an untouched holdout.

Next: prepare the graded presentation/report and demonstrate saved model inference using tools/demo_deep_rl.py, including its missed-positive case. Use saved learning curves and RL-only comparisons; disclose different historical tabular features/training data and unstable recent labels. No manual source downloading is required. Future model experiments need a new declared validation/test design.

The following feasibility and original designs are preserved historical background and do not override the active RL-only scope.

## Preserved feasibility plan, 11 October 2026

The source-archive census was completed and had insufficient 2025 SKIPPED coverage. The preceding feasibility stage used an API-first 2025 audit. Run `tools/collect_2025_feasibility.py` from the repository root to collect or resume `api-2025-feasibility-001` in its separate data directory. Its frozen manifest selects one standard Div. 2 contest uniformly from each two-month block with seed 736, without inspecting outcomes. Preserve all raw participation types; describe in-window individual CONTESTANT, OUT_OF_COMPETITION and mixed sessions separately. `cf_pilot/outcomes.py` distinguishes replacement-compatible skips, skipped final attempts and all-skipped patterns. These are observed descriptions, not reasons, AI-use findings or verified misconduct labels. Training remains gated on the outcome/eligibility audit. No additional manual source downloading is required.

The earlier source-archive feasibility and evaluation design is preserved in [HF_BENCHMARK_PLAN.md](HF_BENCHMARK_PLAN.md) as background. That plan applied a 2025-only boundary to the earlier feasibility stage. The user has since explicitly authorized 2026 for the active benchmark above. The plan corrects the earlier assumption that absent archive skips establish negative sessions and treats SKIPPED as a mixed-reason observed verdict. Manual source downloading is no longer required. Verified misconduct labels remain unknown; any observed-outcome benchmark uses a separate versioned table. Its RL acquisition costs are explicitly simulated evidence-group costs. The remaining sections retain the original project design and frozen collection strategy as background; do not silently mutate contest-002.

## Earlier 2026 collection design

The following describes the frozen contest-002 checkpoint, not the active 2025 sample. Analyse completed Codeforces participant-contest sessions. That sampling strategy collects a complete target contest, then the 20 preceding comparable contests completed strictly before it, within 180 days. College and country are not collected. Each target session is available in full. Output is a recommendation for human review of that session, not a permanent judgement about the participant. The original college pilot is preserved as pilot-001.

The target is the latest completed standard CF contest matching Div. 2, at least seven days old, selected without inspecting suspected behaviour. Earlier rounds match the name filter; combined Div. 1 + Div. 2 and explicitly unrated rounds are excluded. Duration may vary and is recorded for later timing normalization. This is a transparent comparability rule, not proof of equal difficulty. A fixed target ID may override recency while retaining maturity checks. The full raw submission census includes excluded participation types; eligibility is checked locally. Selection policy is versioned as standard_div2_v1.

Historical coverage is measured for every eligible target participant: regular means at least five of the 20 earlier rounds, limited means one to four, none_observed means zero in this panel. No-history does not imply a new account. A deterministic development sample of up to 1,000 uses group weights 70/20/10 with shortages redistributed. Record per-group inclusion probabilities and preserve everyone outside the sample. These weights prioritize history-rich development without estimating misconduct prevalence.

Archive all earlier raw pages; store compact participation facts for the target population, then normalize earlier submission sequences only for the development sample. No per-user calls are required for this first expansion. Training still requires evidence labels and multiple target contests with frozen temporal and participant partitions.

## Work order

1. Resumable public API collector and session-integrity tests.
2. A small real-data pilot and an explicit coverage audit.
3. Evidence register and validated positive/presumed-negative/unknown session labels.
4. Timing/attempt/history features and frozen temporal/user partitions.
5. Rules, logistic regression and tree-based supervised baselines.
6. RL environment for acquiring earlier evidence and ending with referral/no referral.
7. Q-learning, SARSA and DQN comparison under matched acquisition budgets.
8. Five-seed evaluation, cost/error sensitivity, failure cases and demonstration.

## RL formulation

Episode: one completed participant-contest session. Initial observation: full within-contest aggregates. Actions: retrieve earlier rating history, retrieve earlier submission history, refer, finish without referral. Retrieved history predates the target contest. All evidence is cached during training and hidden until acquired. Cost reflects recorded API requests and is evaluated under cold and warm cache conditions. There is no artificial cost for inspecting already-available target submissions.

Tabular agents use compact discretized observations. DQN uses a small numerical network. Invalid repeated acquisition actions are excluded in selection and target calculations. Terminal decisions end the episode. A full-feature classifier and acquire-all policy establish whether RL adds value.

## Data and labels

Keep raw responses, normalized records, feature tables and labels separate. Record acquisition time, parameters, checksums and version. Unknown/unflagged does not imply verified fair participation. Positive labels apply only to the specific session and misconduct established by independent evidence. Evidence of copying does not by itself establish AI use. Presumed negatives retain an explicit confidence category.

Administrative outcomes used to establish labels cannot become model features: omit sanctions, disqualification, related rating adjustments, missing-standings proxies, evidence text and identifiers. Verdict features require an audit for administrative changes. Start with timing/attempt transitions and pre-target history.

## Training checks

Whole sessions stay within one partition. Main test uses later contests; an additional test holds out users. Fit transformations on training data. Tune on validation data. Freeze the final test manifest before tuning. Synthetic variants remain with their source session. Unknown sessions support output demonstrations/referral rates, not correctness metrics.

Before trusting training: verify label provenance, split integrity, feature allowlists, no future-history leakage, correct observation masks, terminal rewards and shuffled-label performance. Repeat five seeds; report variation. Balanced training batches do not establish deployment precision.

## Results

Report precision/recall against the declared benchmark labels, false-positive rate explicitly against presumed negatives, referral rate, requests per session, held-out reward and comparisons against no-history/full-history/threshold baselines. Report limitations of noisy labels and selection. DQN values are policy preferences, not calibrated cheating probabilities.

## Assessment documents

The user supplied LAB CA-Rubrics.docx and LAB CA-Report Format.docx in D:\Downloads. The rubric allocates 5 marks to formulation, 10 to implementation, 10 to results/analysis. The report asks for state/actions/reward/policy/discount/termination, architecture, hyperparameters, training, demonstration, graphs, limitations and references. Multiple RL algorithms are retained from the user's course requirement.
