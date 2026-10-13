> Historical record. The current scope and formal report are in [the main README](../../README.md). This document does not override the active RL-only experiment.

> Historical experiment. The active RL-only model and fresh evaluation are documented in [DEEP_RL.md](../DEEP_RL.md). The earlier test contests below are now development data in the new version.

# Recorded outcome training experiment

The trained task is predicting an all-skipped, multi-problem recorded outcome from behavioural summaries. Neither the target nor a review recommendation establishes cheating or AI use. The immutable benchmark and eligibility definitions are in BENCHMARK.md.

## Run and saved artifacts

Training dependencies are listed in requirements-training.txt. The collection and benchmark builders remain standard-library tools. If training dependencies are absent, install them in the intended Python environment before running the training command. They were already installed for this experiment; no package installation was needed.

Run `python tools/train_benchmark.py` from the repository root. The entry point limits BLAS threads to one. config/training-2026.json freezes the settings. Models are saved as inspectable JSON under data/training-2026-002; results, curves, account-session predictions and anonymized examples are under reports/training-2026-002. Raw data, models and account lists stay outside Git. A completed rerun verifies saved artifact hashes and returns results without retraining. Use a new run ID for a changed experiment.

Run `python tools/plot_training.py` after training to render the saved results. Plots do not choose models or change parameters.

## The environment

One episode is one already completed participant-contest session. This is a partially observed, finite evidence-inspection simulator. The agent can request cached summaries or terminate with LOW_PRIORITY or REVIEW. Actions change what the agent observes, not the contestant's behaviour or platform outcome. The target stays hidden from observations and is used only for reward and subsequent evaluation.

The initial observation contains training-fitted bins of attempts, attempted problems and the relative time of the last submission. Additional actions reveal one compact summary each: median submission gap for pacing, retry fraction for trajectory, or distinct-language count. Each numeric value is discretized using training quartiles; duplicate boundaries may produce fewer occupied bins. Bin 4 means a missing value. The value -1 means evidence not yet inspected. A maximum of three inspections is allowed; terminal actions are always available, repeated inspection is masked, and an episode has at most four actions. Remaining budget is determined from the observation mask.

This compact six-summary representation keeps tabular learning feasible. Q-learning and SARSA use identical observations and costs. The matched logistic baseline uses the same six summaries as continuous values with missing indicators, so representation still differs. The full logistic baseline uses all 24 behavioural features and their missing indicators; it has more information than the tabular agents. Those differences must remain explicit in the comparison.

## Algorithms and settings

Q-learning updates toward reward plus the maximum valid next-action value. SARSA updates toward reward plus the value of its actual next action under the current exploratory policy. Terminal transitions use only the terminal reward, with no bootstrap. Invalid actions are excluded in both selection and updates.

Each algorithm trains five seeds: 42, 736, 2026, 13 and 2718. Every seed receives 500,000 episodes sampled uniformly with replacement from the natural training session distribution. Both algorithms receive the same session sample sequence for a given seed, with a separate action random generator. Learning rate is 0.05 and gamma is 1 for the short finite episode. Epsilon falls linearly from 1 to 0.05 over 400,000 episodes and then remains at 0.05. Greedy validation runs every 50,000 episodes. Each seed's saved policy is its best validation-reward checkpoint, not necessarily its final training state. Inference is deterministic with no exploratory actions; unseen states default to low priority.

Logistic regression uses the frozen training-only normalization and fits natural class frequencies without class balancing or oversampling. C is selected from 0.1, 1 and 10, with an LBFGS fit capped at 500 iterations. A convergence warning stops the experiment. Thresholds are chosen on validation to maximize the declared reward, including a candidate making no referrals and grouping tied scores together. Coefficients and intercepts are serialized as JSON; no executable model pickle is required. Reference: [scikit-learn LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The simple timing rule inspects pacing and trajectory summaries and refers when the median gap is below a threshold and retry fraction above another. Candidate thresholds come from training quartiles and are selected using validation reward. Always-low-priority and inspect-all matched/full logistic policies provide practical comparisons.

## Declared reward and cost assumptions

| Terminal result against the recorded outcome | Reward |
| --- | ---: |
| Review a positive outcome | 20 |
| Miss a positive outcome | -20 |
| Assign low priority to a comparison | 1 |
| Refer a comparison | -2 |

Each additional inspection costs 0.05 reward units. The larger miss penalty represents a chosen review objective for this prototype. These coefficients are assumptions, not measured human-review times, platform sanctions or actual network costs. Main conclusions are conditional on this fixed cost model. Logistic references inspect all three additional groups and incur 0.15; the timing rule inspects two and incurs 0.10. The RL policies can stop earlier. These are simulated costs even though all summaries are already cached.

## Selection and evaluation

Whole-contest training, validation and final-test partitions remain frozen. Models, thresholds and checkpoints use only training/validation data. A selection file freezes the deployment policy and all model hashes before opening final-test CSVs. The chosen deployment policy is the RL seed/checkpoint with maximum validation reward across the ten saved candidates, with stable ordering for exact ties. The final report also includes all five seeds for each algorithm, with mean and sample standard deviation, rather than presenting a single favourable seed as algorithm performance.

Evaluation reports proxy precision, recall, F1, referral rate, confusion counts, average inspections and reward. Results are broken out by contest, current participation stratum, and observed handles unseen during training or during both training and validation. The primary temporal split includes returning handles and is not a fully user-disjoint experiment. Test errors on the comparison class do not identify verified innocent people.

Learning curves are recorded results. A non-monotonic curve or stronger classifier is a valid finding; convergence and RL superiority are not assumed. Account-session output names the contest, recommendation, actual recorded outcome and inspected summaries. Handles are reporting identifiers only. A policy value is not a calibrated cheating probability.

The original training-2026-001 fit is preserved. A reporting performance correction caches the previously-seen-handle union once. training-2026-002 reuses all original model weights and the identical validation selection, recorded in model-provenance.json. No model retuning followed test access.
