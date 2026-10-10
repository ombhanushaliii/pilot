# Post contest submission review project

## Fixed scope

Analyse completed Codeforces participant-contest sessions. The active sampling strategy collects a complete target contest, then the 20 preceding comparable contests completed strictly before it, within 180 days. College and country are not collected. Each target session is available in full. Output is a recommendation for human review of that session, not a permanent judgement about the participant. The original college pilot is preserved as pilot-001.

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
