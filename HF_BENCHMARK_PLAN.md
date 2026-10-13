# Feasibility and benchmark plan for archived Codeforces submissions

Status: planning and bounded feasibility inspection. No model training is authorised by a successful download alone. The next deliverable is a measured go/no-go report. No manual source downloads are required.

## Binding date restriction

The user requires 2025 submissions only. Use UTC creation timestamps in [2025-01-01, 2026-01-01), or [1735689600, 1767225600). Training, validation, testing, historical-context features and source-comparison pools must all obey this restriction. No 2024 warm-up history and no 2026 transfer test are part of this benchmark. Retain existing older/newer archives only as prior work and software fixtures.

The target contest must also take place and finish within 2025, and the submission must belong to its eligible contest-time session. A 2025 practice submission to a 2018 problem does not turn that old contest into a 2025 contest. Check both contest metadata and submission timestamps.

The HF revision dates to May 2025. It can at most support part of the requested year, and its actual submission coverage may end earlier. First measure its 2025 subset. If it fails, the fallback is automated API collection of 2025 contests, not extending the year range. The date restriction improves period relevance but does not establish why a verdict is SKIPPED.

## 1. What we are trying to establish

Determine whether open-r1/codeforces-submissions provides enough usable SKIPPED examples, source coverage and complete API-linked contest sessions to support a reproducible post-contest benchmark. If it does, compare behavioural/source baselines and RL evidence-acquisition policies. The evaluated target is an observed verdict outcome at a declared snapshot, not verified AI use or a permanent judgement about a person.

Two corrections to the earlier plan are essential:

- No SKIPPED row in the HF archive is not a negative session label. The archive is an incomplete compilation of submissions. Complete API session records must establish both classes.
- SKIPPED does not by itself identify a reason. Official contest rules select particular submissions for system testing. First-hand community reports distinguish ordinary resubmission-related skips from disqualification-related skips. We must measure those patterns and avoid calling the entire target an enforcement label.

Sources: [Codeforces contest rules](https://codeforces.com/blog/entry/4088), [first-hand report about skip reasons](https://codeforces.com/blog/entry/149667). The latter is a community observation, not an official API reason field.

## 2. What we already have

| Asset | Reuse | Limitation |
| --- | --- | --- |
| Public API collector, caching, pagination and checksums | Full older-contest metadata collection | Use a new run; the old config restricts target age |
| 21 complete raw contest archives from 2026 | Reuse collector engineering and integrity checks | Data are outside the required 2025 benchmark |
| 1,000-person development sample and 77-feature timing/history builder | Reuse pure feature functions and tests | Existing report builder assumes one target and sampled users |
| Five manually collected sources | Extraction fixtures | They do not form a training dataset |
| Existing misconduct register | Preserve | All 21,121 labels remain unknown |
| HF default archive | Potential source enrichment | No handles, participant type, row capture time or verified misconduct labels |

Pin HF revision 07e7bf31f284b0e90e02dfc7ab675d5821e1c068, last updated 14 May 2025. It has 36 default shards and 12,591,518 reported rows, about 5.55 GB compressed. Use default only. selected_accepted and selected_incorrect deliberately select solutions and distort attempt coverage.

Live viewer statistics cover only 4,197,180 rows and report partial=true. Within that portion there are 13 SKIPPED and 4,173,687 OK rows. This is strong evidence of collection skew in the indexed portion, not a full-archive SKIPPED count. Never extrapolate it as a prevalence estimate. [HF partial-statistics documentation](https://huggingface.co/docs/dataset-viewer/statistics)

A bounded all-year planning scan completed 8 of 36 shards, accounting for 2,798,120 rows and 13 distinct SKIPPED IDs. Those positives belong to contests 1004 and 1241 in 2018/2019 and are excluded by the new requirement. The run stopped at its time budget; it was not a full census. A separate 2025-only audit uses timestamp bounds before reading candidate verdict columns.

The observed default schema lacks both original_code and og_source. selected_accepted has og_source despite the card naming original_code. Preserve available bytes and provenance; do not promise exact original uploaded code. Dataset metadata and card prose also disagree about the licence name; retain attribution and record the discrepancy.

## 3. Stage A: complete the cheap census before downloading sources

Freeze a manifest containing revision, file paths, expected row counts and audit code version. Inspect Parquet footers and project only needed metadata columns. Record completed shards and fail visibly on incomplete requests. Range responses must be checked; a server ignoring Range must not trigger an accidental multi-gigabyte download.

Count 2025 rows and unique submission IDs; 2025 SKIPPED rows and unique IDs; contest/month/language distribution; null or invalid IDs/timestamps; duplicate conflicts; and concentration in a small number of contests. Footer metadata may be inspected to exclude out-of-window groups; their actual submissions must not enter the benchmark. A row count is not a count of independent people or incidents. Preserve row/shard references for selected records.

The first audited shard has 350 small row groups, so indiscriminate remote column reading can produce many HTTP requests. Use footer statistics to exclude groups that cannot contain SKIPPED; read necessary columns only for candidate groups. Compare measured request cost with downloading a bounded selected shard. Cache completed audit outputs. Do not load the archive into pandas.

Initial resource policy: one network worker, bounded batches, checkpoints, 1 GB working-memory target, and an explicit download/time budget per run. Report measured RSS, disk and transferred bytes. Pause before a substantial full download and present the measured requirement. A planning probe has its own smaller budget. No contestant code is executed.

Deliverable: a complete-versus-partial census report, positive ID inventory and candidate contest table. If the entire archive has only a few dozen SKIPPED IDs, it is unsuitable as the main positive training source. Oversampling does not create independent evidence.

## 4. Stage B: reconstruct a small automated feasibility cohort

Choose about six eligible 2025 contests across the covered portion of that year with a frozen seed and declared contest-format policy. Avoid selecting only contests with the largest positive counts. Include ordinary coverage-selected contests and, separately, a few positive-enriched diagnostic contests. Enrichment is for inspection, never a population estimate. If the archive covers fewer than six suitable contests, report the limitation instead of quietly including an earlier year.

For each chosen contest, download complete contest.status pages and contest metadata through the existing rate-limited collector. Preserve all participation types initially. Do not silently apply the old OUT_OF_COMPETITION exclusion: audit whether that current category disproportionately removes positive candidates. Use timestamps, team/ghost status and available participation evidence to define eligibility. Ambiguous original participation stays in a separately reported stratum. PRACTICE and VIRTUAL sessions are not silently pooled with official contestants.

Join HF rows to API submissions by numeric submission ID. Validate original contest ID, problem index, creation time and language, with explicitly documented language-name normalisation. Canonical problem_id may represent an alias and is not the identity key. Quarantine identity mismatches, conflicting duplicates and unresolvable rows. Do not silently coerce missing or nonintegral IDs.

Keep hf_verdict and api_verdict separately, with HF revision and API acquisition time. Publish their agreement table. Current API results can reflect later rejudging or reversals; disagreement is not automatically a parser bug. Do not choose the snapshot that produces more positives.

Define the primary outcome from a complete frozen API session snapshot:

- 1: at least one eligible submission currently has SKIPPED.
- 0: the complete eligible session has no SKIPPED and no unresolved/pending record that could alter this decision.
- Unknown/excluded: incomplete pagination, unresolved eligibility/identity, or unstable/incomplete outcome records.

Maintain these in a new observed-outcome table. Do not replace the independent misconduct register. For archive-only row experiments, label the task row-level archived-verdict prediction and do not claim complete behavioural histories.

## 5. Test whether SKIPPED is useful for our intended question

For each positive session, automatically describe where skips occur: earlier attempts with later submissions on the same problem; whether the final attempt is also skipped; whether skips affect several problems; and whether every observed attempt is skipped. These are diagnostic categories, not inferred reasons or new truth labels.

Compare these categories by contest format, date, participation type and language. Check whether a simple resubmission-pattern baseline explains most positives. If the useful signal is primarily ordinary replacement behaviour, a high score does not demonstrate misconduct review value. Retain the narrow verdict-prediction claim or stop the cheating-related interpretation.

For the coverage audit, inspect up to 100 positive and 100 randomly sampled outcome-negative sessions across the selected contests, using all available positives if fewer exist. Selection is automated. No user source inspection is required. Record sample probabilities. This balanced diagnostic set is not the validation/test set.

Measure separately by outcome and contest:

1. Submission-ID match rate and identity-conflict count.
2. Sessions with at least one available source.
3. Fraction of each session's attempts with source.
4. Consecutive same-problem attempt pairs with both sources.
5. Unsupported languages and potentially altered source provenance.
6. HF/API verdict agreement and eligibility exclusions.

An edit between attempts 1 and 4 is not an adjacent retry edit when attempts 2 and 3 lack code. Store the gap explicitly or leave the adjacent-pair feature missing. Never use zero to mean missing code.

## 6. Freeze selection and splits before predictive experiments

The final population is complete API-derived sessions from a predeclared chronological window within 2025. Historical windows stop at 1 January 2025 and never reach into 2024. Include sparse-history participants with missing-history flags; do not require regular activity to obtain a clean-looking sample. Keep organisation and country out.

If the population is manageable, retain every session. Otherwise use a deterministic uniform session sample with recorded probabilities. Use class weights or positive oversampling only inside training; retain observed outcome prevalence in validation and testing. For a separate case-control diagnostic, match by contest and optionally pre-contest experience, not by a feature we are trying to test such as retry speed.

Plan chronological training/validation/test blocks, with earliest contests supplying historical context. Aim for multiple later test contests, not a single final round. A 60/20/20 allocation of eligible modelling contests is a starting point, adjusted for dates/counts before any model scores are seen. No row-wise random split of attempts or source pairs.

For a strict participant-disjoint experiment, reserve handles whose labelled sessions never enter fitting or validation. Their earlier unlabelled history may be used as evidence at test time, but cannot fit normalisation, bins, vocabularies or reference models. Public handles can change, so report known alias handling and residual person-level ambiguity.

Audit identical and near-identical code and reused problem aliases across partitions. For the temporal test, never move future records into training to keep a cluster together. Report an additional decontaminated test or quarantine cross-boundary duplicates. Preserve the natural temporal test as a separate result.

## 7. Features and leakage protection

Start with target attempt counts, retry timing, submission spacing, language changes and strictly prior behavioural medians/deviations. An inter-submission gap is elapsed submission time, not verified coding or AI-generation time.

Add source features only after coverage passes: code/token length, token edit ratio for true adjacent pairs, and template-aware same-problem similarity. Use static language-aware processing. Strip obvious identifiers and source-origin comments, cap pathological input sizes and record unsupported syntax as missing. Shared boilerplate or a common correct algorithm is not proof of copying.

Exclude all verdicts, passed-test counts, runtime/memory results, final scores/ranks, acceptance-derived totals, sanctions, rating adjustments, IDs, handles, URLs, shard identifiers and evidence notes. Administrative data used to derive or investigate outcomes remain outside model observations.

Also exclude current user ratings and retrospective problem ratings from the first experiment unless their availability at the declared review cutoff is demonstrated. Current snapshots can contain information that did not exist at historical review time. Audit historical eligibility for the same issue; a later administrative change must not silently alter what is presented as an earlier behavioural baseline.

Fit imputation, scaling, bins, vocabulary and learned references on training only. Run a missingness-only model and a dataset-origin/language diagnostic to expose coverage shortcuts. Compare metadata-only and source-enriched methods on the same sessions, including source-missing cases. Any source-complete-only result must be a secondary restricted-population experiment.

Within-contest code similarity may use unlabelled peers from the complete target contest because review occurs after it ends. Document this as batch information available at review time, not as training on test labels.

## 8. Baselines before RL

Run never-refer, always-refer, a simple behavioural rule, the resubmission-pattern diagnostic, logistic regression and a small tree model. Compare target-only and full-information versions. Tune thresholds on validation only.

A predictive resubmission baseline may use repeat-attempt counts and order, but not final verdicts. The retrospective analysis of skip locations may use labels to explain errors; it is a diagnostic, not a deployable competing model.

Check whether metadata beats prevalence/random ranking at the declared review budget across several validation contests. Then test whether source features add useful performance on the same population. If they add no reliable value, omit them from the primary RL environment rather than making source collection mandatory.

## 9. Small, explicit RL experiment

An episode is one completed participant-contest session. The agent initially sees complete target timing aggregates and availability flags. It can acquire earlier behavioural context, acquire source summaries if available, refer, or finish without referral. Each evidence group is acquired at most once; terminal actions end the episode. At most two acquisitions plus a decision keeps the experiment small.

The hidden environment record contains the remaining feature groups and outcome. The learner sees only its current observation and acquisition mask. This is a partially observed information-acquisition task; discretised summaries are an approximation, not a claim that they completely describe a person's behaviour. The environment never passes the outcome through reset/step observations or auxiliary fields consumed by the learner.

Use Q-learning and SARSA first, followed by a small CPU DQN. The primary algorithm comparison uses the same compact information and training-fitted bins. A continuous-input DQN is a separate representation experiment. Avoid training-set classifier scores as states unless produced out of fold.

Keep the tabular state small: a few predeclared timing/attempt summaries, one or two acquired-history/source summaries, missing-value categories and the acquisition mask. Use roughly three bins per numerical summary, fitted on training data. Calculate the resulting state count before training; do not form a table from the Cartesian product of all 77 features. Start DQN with a small network rather than treating a large model as necessary. Record the final architecture, learning rate, exploration schedule, replay/target-update settings and seeds in a frozen experiment config.

A starting reward specification is TP +1, TN 0, FP -1 and FN -2, with a simulated cost of 0.05 per acquired group and discount 1 for this short finite episode. Choose the final tradeoff on validation and report sensitivity, including zero acquisition cost. These are declared experimental preferences, not measured network savings. The terminal label is the frozen observed outcome.

Compare acquire-all, acquire-none, random acquisition and a fixed acquisition order using identical terminal rewards and costs. Give Q-learning, SARSA and DQN matched environment-step budgets, five seeds and similarly small tuning budgets. Report wall time too. RL need not win for the comparison to be a valid course result.

## 10. Tests and reported results

Before training: all retained joins validate; session counts reconcile; outcome tables cannot enter feature construction; changing verdict/test-resource fields leaves predictor values unchanged; adding future records leaves earlier features unchanged; whole sessions and required groups stay within their partition; transformations fit only training.

For RL: acquired data stay hidden beforehand; masks affect exploration, action selection and bootstrapping; repeated acquisitions are impossible; missing sources do not produce fabricated values; terminal steps do not bootstrap; reward tests cover every decision/outcome pair. Use a shuffled-outcome control and a deliberately leaked diagnostic to ensure the audit detects leakage.

On untouched natural-prevalence test contests, report precision and recall at 1%, 5% and 10% review budgets, PR-AUC for a declared ranking score, false-referral rate among outcome negatives, acquisition count, return, runtime and memory. Q-values are not cheating probabilities. If a policy supplies only a binary decision, report its operating point rather than inventing a PR curve.

Show per-contest results and uncertainty by resampling contests or another appropriate independent group. Five seeds measure optimisation variation; they do not replace sampling uncertainty. Report language/history/skip-pattern breakdowns and failures. Do not tune on final test results and keep calling that test untouched.

## 11. Go/no-go rules

| Gate | Proceed when | Otherwise |
| --- | --- | --- |
| Census | All 36 shards accounted for; exact row count reconciles; unique positive inventory available | Report partial coverage and resume; do not infer zero from failed queries |
| Identity | Every retained source/API join validates; conflicts are quarantined | Fix joins or restrict the declared population |
| Session outcomes | Complete API records and explicit eligibility; both snapshots preserved | Keep incomplete/ambiguous sessions out of binary evaluation |
| Positive support | Planning targets of about 200 training, 50 validation and 100 test positive sessions across multiple contests and users | Expand a declared chronological window once, or use a smaller exploratory study without stable-accuracy claims |
| Source usefulness | Adjacent-pair coverage and class/language missingness permit the intended feature; source model adds validation value | Use metadata as primary, source metrics as optional or omit them |
| Semantic usefulness | Results remain informative after ordinary replacement-pattern analysis | Narrow the project to verdict prediction; do not claim cheating detection |
| RL usefulness | Comparable accuracy/review yield with fewer acquired groups, or a clearly measured tradeoff | Report that full-information supervised learning is preferable on this benchmark |

The numerical positive targets are practical planning targets, not a guarantee of statistical power. Correlated copies, users and contests reduce effective sample size. If there are only a handful of independent incidents, more epochs or duplicate samples do not solve that.

## 12. Work sequence and deliverables

1. Complete census and feasibility tables; freeze the go/no-go decision.
2. If viable, build a new older-contest API/source benchmark with explicit outcome and split manifests.
3. Run metadata and source baselines; freeze the RL environment and comparison settings.
4. Train Q-learning, SARSA and DQN; run leakage checks, ablations and final tests.
5. Produce learning curves, comparison tables, a post-contest review demo, reproducible commands and the course report.

Reuse the working collector and feature code. If the HF archive fails the support/coverage gate, collect complete API populations from 2025 for a separately named metadata benchmark and retain this archive only where its 2025 sources help. Existing 2026 records do not enter training, validation or testing. The same ambiguity of SKIPPED reasons must be acknowledged in that fallback.

No finite plan can eliminate every mistake. The point of these gates is to make incomplete collection, ambiguous labels, leakage and failed assumptions visible before they produce impressive but misleading results.
