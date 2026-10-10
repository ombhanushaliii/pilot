# Post contest submission review project

## Fixed scope

Analyse completed Codeforces participant-contest sessions. A college leaderboard selects an initial cohort; organisation is not a model input. Each target session is available in full. Output is a recommendation for human review of that session, not a permanent judgement about the participant.

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
