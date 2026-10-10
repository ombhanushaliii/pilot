# Project status

Milestone 1 completed on 10 October 2026: the collector, session builder and integrity tests are implemented in D:\pilot.

## Verified results

- Source: https://codeforces.com/ratings/organization/5276
- Organisation table contains 126 handles. Sidebar/global profiles are excluded.
- Reproducible sample: 24 handles, seed 736.
- Selected completed contests: 2176, 2188, 2192.
- Archived successful responses: 128, with original-byte checksums.
- Normalized unique submission records: 7,341 across target and historical queries, including retained practice records.
- Rating records: 379.
- Archived standings rows keyed by individual handle: 34,007.
- Participant-contest candidate combinations: 72.
- Eligible official target sessions: 6, containing 20 submissions. All six have earlier contest context.
- Excluded candidates: 64 with no returned submissions; 2 with practice-only activity. These are not negative training examples.
- All 24 historical paginations exhausted the available user history within the configured cap. No history boundary heuristic was needed in this pilot.
- Failed API attempts: 0. Snapshot conflicts: 0.
- 21 automated tests passed, including the full fake-API workflow and offline rerun.
- Actual live rerun: 0 network requests, unchanged record counts, unchanged response manifest and unchanged manual label worksheet.
- Observed minimum interval between request starts: 2.100078 seconds.
- Final collection audit passed with no integrity issues.

## Files and commands

Code: cf_pilot/. Config: config/pilot.json. Database: data/pilot.sqlite. Responses: data/raw/pilot-001/. Reports: reports/pilot-001/.

Run python -m unittest discover -s tests -v for tests. Run python -m cf_pilot collect --config config/pilot.json to resume/reuse the frozen dataset. Run python -m cf_pilot check --run-id pilot-001 for a fresh integrity audit.

## Contest strategy implementation

The user requested a Git checkpoint before continuing. Created repository D:\pilot and committed the original collector/docs/tests as 9c2df27 (Checkpoint tested Codeforces pilot collector). Global SSH commit signing failed; this checkpoint used a command-local signing override, without changing Git settings. Generated data and reports remain local and ignored by Git.

Implemented a second sampling strategy: full target-contest submission census, 20 preceding standard CF Div. 2 rounds within 180 days, historical participation coverage for all eligible target participants, and a reproducible development sample of up to 1,000. College/country are not collected. The development sample uses 70/20/10 history-coverage weights, with shortages redistributed and inclusion probabilities recorded.

33 automated tests passed after the selection and paging fixes. The original pilot-001 still passes its live-data integrity audit with unchanged counts. Contest pagination is tested for clipped short pages, missing boundary anchors, added practice submissions, interrupted historical collection, exclusions, future context, population/sample completeness, and offline reruns.

Live run contest-001 failed before downloading target submissions because identical-duration matching could not supply 20 earlier rounds. Its failed state and first response are retained. The revised standard_div2_v1 policy permits varying durations and excludes combined Div. 1 + Div. 2 / explicitly unrated rounds. Duration will need to be accounted for in later timing features.

Live run contest-002 completed and passed its final audit: target 2269 (Codeforces Round 1124, Div. 2) and its 20 preceding comparable rounds. All 21 contest jobs exhausted their ID-bounded snapshots. Full archived traversals contain 2,306,696 distinct submission records across these contests, including practice and other excluded participation.

The target snapshot contains 77,434 submission records, 21,121 individual-author candidate sessions and 10,344 eligible official participants with 41,942 official submissions. New practice submissions shifted a historical offset page during the first attempt. Collection failed visibly, then resumed after an ID-anchor realignment fix. The previous boundary must exist in the response; prefixes are deduplicated and missing anchors still fail. Each traversal preserves the snapshot bounded by its first submission ID. No misconduct labels have been inferred from missing standings or activity patterns.

Final coverage: 4,059 regular participants (at least five earlier rounds), 4,856 limited-history participants (one to four), and 1,429 with none observed in the panel. Development sample: 1,000, comprising 700 regular, 200 limited and 100 none_observed. Coverage and sampling probabilities are reported for all 10,344 eligible participants. Earlier account history outside the panel is not claimed complete; skill stratification and per-user rating retrieval are not yet implemented.

Archived successful responses: 285. Three transient HTTP/API attempts failed and were retried successfully. Recorded page realignments: 5. Conflicting normalized snapshots: 0. Minimum observed request-start interval: 2.100101 seconds. Normalized submission records: 107,666, comprising the complete target snapshot plus eligible earlier submissions for sampled participants. Earlier raw pages are preserved in full.

Actual completed-run rerun: zero network requests or attempts added; unchanged table counts, response manifest, coverage CSV, development sample CSV and label worksheet. Verification evidence: reports/contest-002/rerun_verification.json. The final collection audit passed with no issues. All 21,121 candidate labels remain unknown; no models have been trained.

Reports: reports/contest-002/. Commands: python -m cf_pilot collect --config config/contest.json and python -m cf_pilot check --run-id contest-002. User-facing result, audit, development sample and rerun evidence were exported to C:\Users\Om Bhanushali\Documents\Codex\2026-10-10\pro\outputs.

## Next work

On 10 October 2026, recorded the verified sample-level inventory in DATA_INVENTORY.md and specified the next milestone in PRETRAINING_PLAN.md. The 1,000 sampled participants have 4,286 official target submissions and 30,232 eligible earlier submissions (34,518 official records total).

Implemented and verified the feature/evidence milestone by 11 October 2026. FEATURES_AND_REVIEW.md documents definitions and commands. cf_pilot/features.py builds timing_panel_v1 offline after the full collection audit, with 1,000 session rows and an explicit 77-column development allowlist: 15 base features, 2 history indicators and 60 prior-reference/deviation columns. Identifiers, labels, demographics, standings and administrative outcomes are excluded from predictors. Verdict snapshots are separate exploratory diagnostics. Context uses only eligible frozen-panel contests finishing strictly before the target, with at least five earlier usable values per reference. Missing and zero-IQR cases remain missing rather than fabricating values.

Real build used exactly 34,518 eligible submissions. 424 target sessions have no same-problem retry; 300 participants have fewer than five earlier panel sessions. Some scaled-deviation columns are almost entirely missing or uninformative and must be assessed using training-only feature selection later. No imputation/scaling has been fitted.

Implemented deterministic random review export, context navigation and validated evidence import. reports/contest-002/evidence_register.csv contains 100 pending unknown sessions from the weighted development sample; its companion context JSON contains official submission references without a suspicion score. Existing worksheets and registers are preserved. Import defaults to dry run, checks the whole file before applying, distinguishes presumed negatives from substantiated session misconduct/AI misuse, validates submission ownership, reviewer dates and declared evidence, retains append-only review events and supports unknown reversals. It checks structure, not the truth of external sources.

50 automated tests passed, including collection regressions, hand-calculated features, timestamp ties, sparse/zero-IQR histories, future/practice exclusion, verdict-independence, frozen artifacts, review atomicity, idempotency, reversals, provenance auditing and reference ownership. Actual real-data rerun added zero network attempts, produced identical feature files and retained collection counts, labels, raw manifest and legacy worksheet. Re-export preserved the register. Evidence: reports/contest-002/feature_rerun_verification.json. The 100-row register passed dry-run validation as 100 untouched unknown rows, with zero review events. All 10,344 eligible target labels remain unknown; no real labels have been applied and no model has been trained.

Commands: python -m cf_pilot features --run-id contest-002; python -m cf_pilot review-export --run-id contest-002 --limit 100; python -m cf_pilot review-import --run-id contest-002 --input reports/contest-002/evidence_register.csv; python -m cf_pilot review-check --run-id contest-002. Use --apply only to explicitly record completed reviews. User-facing feature matrix, manifest, register, context, guide and rerun evidence are exported to the existing Codex outputs directory.

The immediate next task is the review-feasibility audit: inspect random sessions, seek independently substantiated session-specific findings, record unresolved cases, and establish usable class support. Multiple target contests and frozen evaluation partitions still follow before training. Do not replace missing outcomes with timing-based labels or suspicion-score rewards.

This original random leaderboard sample has thin participation in the selected rounds. The completed contest strategy supersedes its sampling. Preserve pilot-001 as the tested engineering snapshot. Next establish independent session-label evidence and an explicit feature allowlist, then repeat across additional target contests for temporal and participant-held-out evaluation.

No misconduct labels or models have been inferred/trained. All labels are unknown. The first feature allowlist is implemented; independent reviewed outcomes and frozen temporal/user partitions remain required. The user's next useful input is independently checkable candidate cases, specified by handle, contest and evidence source; collection itself needs no additional credentials.

Read PROJECT_PLAN.md and README.md before continuing.
