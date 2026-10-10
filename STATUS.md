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

## Next work

This random leaderboard sample has thin participation in the selected rounds. Before training, expand collection to a more active cohort and more finished rounds, and add a comparison population. Six target sessions are sufficient to exercise the collector, not model training. Preserve pilot-001 as the tested engineering snapshot.

No misconduct labels or models have been inferred/trained. All labels are unknown. Evidence-backed session labels, feature allowlists and frozen temporal/user partitions remain required. The user's next useful input is independently checkable candidate cases, specified by handle, contest and evidence source; collection itself needs no additional credentials.

Read PROJECT_PLAN.md and README.md before continuing.
