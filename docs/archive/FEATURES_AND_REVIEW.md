> Historical record. The current scope and formal report are in [the main README](../../README.md). This document does not override the active RL-only experiment.

# Features and evidence workflow

This milestone builds offline, post-contest development data. It does not train a model, generate misconduct labels or establish that anomalous timing identifies AI use.

## Build and reproduce

Run commands from D:\pilot with Python 3.11 or newer. No additional dependencies or network requests are needed.

```powershell
python -m cf_pilot features --run-id contest-002
python -m cf_pilot review-export --run-id contest-002 --limit 100 --seed 736
python -m cf_pilot review-check --run-id contest-002
```

The feature command first checks collection integrity, including raw response checksums. It builds one row per selected eligible participant-contest session and refuses incomplete collection. Output is reports/contest-002/timing_panel_v1/. Existing files must match byte for byte; changed files or source snapshots require a new output and, for changed definitions, a new feature version. It does not modify labels or download anything.

## Feature definitions

Only feature_manifest.json model_columns are permitted predictors. Run ID, contest ID and handle are join keys, never predictors. session_metadata.csv records duration, coverage group and inclusion probability for provenance. Do not blindly train on every numeric column in every report.

| Base feature | Definition |
| --- | --- |
| attempts | Count of eligible individual official submissions within the half-open contest interval |
| attempted_problems | Number of distinct contest/problem-index pairs attempted |
| retries | Sum of attempts after the first on each problem |
| retry_gap_min/p10/median_seconds | Adjacent submission spacing on the same problem, sorted by timestamp then submission ID; p10 uses linear interpolation |
| retry_fraction_le_15s/le_60s | Number of same-problem gaps at most the threshold, divided by number of such gaps |
| first/last_submission_fraction | Seconds from contest start to first/last submission, divided by contest duration |
| attempts_per_hour | Total attempts divided by contest duration in hours, not hours spent coding |
| language_switches | Count of language changes between adjacent submissions on the same problem |
| problem_rating_mean/max | Difficulty over distinct attempted problems with an available numeric rating, each problem counted once |
| problem_rating_missing_fraction | Attempted problems without a numeric difficulty divided by attempted problems |

All eligible submission timestamps contribute regardless of final verdict. Administrative verdict changes cannot affect these predictors. Problem difficulty is retrospective metadata and can be missing. Conflicting difficulty snapshots within a problem fail rather than silently choose a value. Duration normalization does not imply equal contest difficulty.

For each base feature, earlier panel sessions provide a median, interquartile range, target-minus-median delta, and delta divided by IQR. Each reference requires at least five earlier non-missing values for that feature. With fewer values, references/deviations are blank. When IQR is zero, scaled deviation is blank. observed_prior_sessions and history_below_minimum make sparse context explicit. No imputation or population-based scaling is learned here.

Eligible earlier sessions must belong to the frozen panel, start within its lookback and finish strictly before the target begins. Practice, future and overlapping sessions cannot contribute. Coverage must agree with the normalized earlier sessions. Blank CSV cells mean missing, not zero. There are 15 base columns, 2 history indicators and 60 reference/deviation columns, for 77 permitted predictors. These correlated columns are a development allowlist, not proof that all should survive later train-only feature selection.

verdict_diagnostics.json separately contains accepted-problem counts, spacing between first accepted submissions on distinct problems, ordinary verdict counts and adjacent ordinary verdict transitions. Administrative/unknown verdicts are not counted as ordinary and do not create transitions across intervening records. These are retrospective snapshots, not the contestant's original judge-feedback timeline. They are excluded from model_columns pending a separate validity audit.

Feature files contain no labels, country, college, standings presence, sanctions, rating adjustments or reviewer notes. The manifest includes input and artifact hashes, missing-value counts and coverage groups. Input provenance includes the eligible snapshots and frozen contest metadata. Synthetic records used in tests are never mixed into the real dataset.

## Evidence review

review-export creates a deterministic random batch of up to 100 eligible sessions from the weighted development sample. It preserves any existing destination, even if someone has edited it. The random batch is not population-representative and is not yet a suspicious-candidate batch. New exports contain pending unknown decisions; leaving rows untouched cannot overwrite existing labels.

A newly exported register has an accompanying .context.json containing eligible submission IDs, timestamps, problem indices, earlier participation count and direct submission-page links. This provides navigation for review without a candidate score or inferred label. Links are constructed from archived IDs; the pages have not been fetched or independently verified by this command. Existing context files are never silently replaced.

The register columns describe the specific session, source URL and type, source collection and review times, reviewers, confidence, submission references, related-case group, sampling reason, disagreement, reversal and assessment notes. Submission IDs are optional, separated by semicolons, and must belong to the named target session. Dates use ISO 8601 with timezone, for example 2026-10-10T12:00:00+05:30. Review/source collection must follow the completed contest and cannot be future-dated.

Use these categories:

- unknown: no established decision, insufficient evidence, disagreement or reversal.
- presumed_negative: reviewed without an independent misconduct finding. This is not verified fair participation. Use evidence_type reviewed_no_independent_finding and explain the review in notes.
- substantiated_misconduct: independently supported session-specific finding or admission. Allowed source types are platform_finding, organizer_finding and verified_admission. Requires a source URL, collection date, session verification, medium/high confidence and related-case ID.
- substantiated_ai_misuse: the same requirements plus ai_explicit=yes, asserting that the evidence establishes AI misuse specifically. Copying alone is insufficient.

Public allegations and feature-based suspicion can be leads, but cannot become positive labels on import. A second reviewer is optional and must be distinct from the first. Conflicting second decisions, disagreement=yes or reversal=yes require label=unknown. Reviewers must inspect sources themselves: a syntactically valid URL or a claimed evidence type is not proof. The importer validates structure and declared provenance, not the truth of a finding.

Validate an edited register first:

```powershell
python -m cf_pilot review-import --run-id contest-002 --input reports/contest-002/evidence_register.csv
```

This is a dry run. Once reviews are complete, apply explicitly:

```powershell
python -m cf_pilot review-import --run-id contest-002 --input reports/contest-002/evidence_register.csv --apply
python -m cf_pilot review-check --run-id contest-002
```

All rows are checked before changes. A bad row rejects the whole file. Reviewed entries create append-only evidence_reviews events and update the corresponding labels atomically. Repeated identical reviewed rows are no-ops, so replaying an old import cannot reapply a previously reversed outcome. An untouched unknown template row is a no-op. To record a reversal, explicitly review it as unknown with reversal=yes, a reviewer, timestamp and explanation. Prior events remain available. Legacy labels_for_review.csv is preserved; its older columns do not satisfy this new import format.

The review audit counts current eligible outcomes and review events and checks decided labels against their latest stored review. It never declares training ready by counting labels alone. Later work must establish evidence quality, usable class support, related-case groups, additional target contests, fixed evaluation partitions and train-only preprocessing. The next review-feasibility audit should also add independently identified cases and document unresolved cases without forcing a balanced dataset.
