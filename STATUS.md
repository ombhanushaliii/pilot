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

## User-supplied evidence sources, 11 October 2026

Audited the supplied Nebius/Christine pasted texts and public blog/revision pages. Parsed 42 unique Nebius candidate sessions for contest 2211: 24 claimed prompt-trap hits, 2 prompt-specific comment claims, 15 behaviour/style allegations and 1 unexplained submission link. 32 candidates have submission links; 25 have both a prompt-related claim and a link. Added five unverified Round 1033 candidates for contest 2120, giving 47 leads. Every lead remains unknown and lead_not_verified. Local artifacts: reports/source-audit-001/candidate_leads.json and .csv. User-facing assessment and candidate files are in the existing Codex outputs directory. Extracted links and handle uniqueness were checked; source-text hashes retained.

Public rendered statement 2211F specifies modulus 676767677, but the alleged hidden contest-time instruction was not recovered. Two direct submission pages were inaccessible to the web reader, so their code has not been verified. No denial was bypassed. The robot_bobot list remains an unresolved historical source rather than session labels; Mike's request to investigate does not confirm every account. Christine's aggregate charts and community list are background/leads, not session-level ground truth. Country and college stay excluded.

Next prioritize verification of the 25 linked prompt-related claims, including the original statement evidence, source artifacts, exact participant/session mapping and disputed/reversed outcomes. A separately versioned evidence cohort would be needed for Nebius: contest 2211 is combined Div. 1 + Div. 2 and is outside standard_div2_v1. Collect comparisons within its own format and preserve contest-002. Do not let label-supplying trap signatures, source text or sanctions become behavioural predictors, and do not claim that trap-visible cases represent all AI misuse. No new contest collection or label import was performed in this audit. Verified afterward that all 21,121 contest-002 candidate labels remain unknown. No collector/model code changed; no tests were rerun for this source-only audit.

## Playwright source feasibility, 11 October 2026

The user authorized Playwright source collection. Added the separate optional cf_pilot/sources.py workflow, pinned requirements-sources.txt, config/source-feasibility.json, SOURCE_COLLECTION.md and nine source-integrity tests. Existing Playwright 1.51.0, psutil 7.0.0 and native Edge/Chrome are available. No WSL or browser download was used. The API collector remains a standard-library workflow.

The five-entry source manifest contains three unverified Nebius leads (368575830, 368565363, 368548267) and two ordinary unknown-labelled contest-002 sampled records (392176110, 392176177). It checks transport and extraction, not prediction accuracy or cross-format comparability. A manifest freezes on first collection; no successful live source artifacts or source run were created in this attempt.

The collector uses one context/page, a dedicated ignored manual-login profile, serial seven-second navigation starts, explicit identity checks, exact DOM source text, immutable UTF-8 files and SHA-256 metadata, failure outcomes and cache-only reruns. Source acquisition never updates the API database or labels. Process-tree RSS sums and system available memory are sampled, with configurable guard thresholds checked between acquisition steps. Those measurements are not PSS or hard OS memory limits.

59 automated tests passed after adding corrupt-resume reporting. Checks cover author/problem mismatch, missing/duplicate source, access denial, redirected login, source whitespace, corruption, frozen settings, interruption/resume, zero-network cached reruns, cadence, limits, exclusive profiles and memory guard behavior, alongside all previous tests. Real DOM extraction through Playwright remains unverified: the local browser-fixture check failed at browser startup in this execution environment, with crash-reporting helper errors in Edge and Chrome. A documented crashpad-disable flag did not fix startup and was removed. No browser fixture or Codeforces extraction is claimed successful, and steady browser RAM is not measured.

The user ran the login helper from normal PowerShell. They reported a looping Cloudflare human-verification challenge despite manual checkbox clicks. Asked them to stop and close the collector window. No challenge bypass, stealth configuration or automated solving was attempted. Cloudflare's official supported-browser documentation explicitly excludes Playwright from supported production challenge-solving environments: https://developers.cloudflare.com/cloudflare-challenges/reference/supported-browsers/ . Source access is currently blocked; the next feasibility check is ordinary manual viewing of the exact submission in the user's existing Helium session. If accessible normally, a small manually saved code sample can support evidence review. If not, retain public metadata and unresolved labels while seeking an authorized source-data route.

Verified after the reported challenge: database SHA-256 matches the pre-source-work baseline, zero Codeforces source artifacts, no login-ready marker, and the earlier baseline contained 21,121 unknown contest-002 labels. No models have been trained. All source implementation files are currently uncommitted.

## Loaded Helium source export, 11 October 2026

The user confirmed that the first source page opens normally in their existing Helium session. Native computer-use initialization failed twice before any UI operation. No Helium profile was accessed or copied. Prepared a self-contained user-click bookmarklet using the same read-only DOM snapshot extractor and a separate standard-library offline importer, cf_pilot/source_exports.py. tools/build_source_exporter.py builds source-exporter.html and inspectable source-export-bookmarklet.js. config/source-browser-export.json declares the same five exact cases under a distinct run ID; existing source-feasibility settings were preserved.

The exporter downloads only source text and submission identity from a loaded page; it makes no navigation/fetch calls. It is restricted to the five specified cases and checks the visible author/problem/language before export. Offline import rechecks schema, timestamp, URL and expected identity, writes immutable source/hash metadata, preserves unknown labels, and rejects differing duplicate or corrupt files. HTTP status is explicitly unobserved. Original whole-page HTML, cookies and arbitrary JSON fields are not archived. This semi-automated evidence acquisition does not establish unattended scraping feasibility or prediction accuracy.

All 64 tests passed in 35.970 seconds, including five new export/import tests and a controlled Node DOM fixture exercising successful download and wrong-host, out-of-manifest, author, challenge and missing-source failure paths. Generated JavaScript passed node --check. Real Helium DOM/export is still awaiting the first user capture; no real source artifacts were created. Database SHA-256 remained b73ce7dbf0b365b34d23d90aa3e524519a984f2d42e98f6f2d4ae572e5d82abc, matching the pre-source baseline. No models trained or labels assigned. Source workflow remains uncommitted.

Rechecked current official API documentation: URLs must be built from submission IDs, handles live in author.members, contest.status includeSources requires asManager and contest-manager permissions, and user.status includeSources is documented for the requesting user's own account. Ordinary website source visibility does not imply contest-manager API access. Accepted-only filtering would omit important failed attempts and later administrative outcomes, so it is unsuitable for this project's behavioural/evidence collection.

## Saved HTML import and extractor correction, 11 October 2026

The user supplied a saved submission page. The exact cause of the bookmarklet error was confirmed: the # column contains a plain numeric submission ID, without a submission hyperlink. Updated the shared DOM extractor to derive the canonical URL from that explicit ID and the observed contest/problem link. It still rejects mismatched authors, ambiguous rows and unsupported identities. Added a regression fixture for plain numeric IDs, wrong IDs, malformed IDs and external problem links.

Added cf_pilot/source_html.py for standard-library offline HTML import. It reads a bounded UTF-8 file, parses only source/title/table facts, decodes HTML entities and normalizes HTML newlines to LF. It validates the title ID and observed table identity against the frozen source manifest, then archives source text and hashes without executing any scripts or copying the entire page. Saved HTML containing already-rendered line lists fails visibly so it cannot become concatenated code. Browser exports reconstruct pretty-printed line elements and record that extraction method; original uploaded whitespace is not guaranteed.

Imported submission 368575830 successfully: observed author Thinkin, contest 2211, problem E, C++20 (GCC 13-64), 2,693 UTF-8 bytes, source SHA-256 8c7e6f08619d8fb2cdc89cedab1acc0dc5f29f9f8527f91a7e05b19cd383cc09. The page contains an Out of competition notice, which is retained as a page fact, not proof of the original participation category or misconduct. Its capture time and HTTP status are unobserved. The evidence lead remains unknown; no prompt-trap instruction or misconduct finding was established. A repeat offline import returned already_saved and made zero requests. The original saved HTML remains local and outside Git.

All 67 automated tests passed in 13.635 seconds after the parser/extractor changes, including saved-page entities/newlines, title/ID/author/problem mismatches, duplicate source/table rejection, challenge detection and rendered-line safeguards. Generated JavaScript passed node --check. The corrected bookmarklet is regenerated in outputs and work under the visible v2 name; it still needs a successful live browser export. Raw Codeforces metadata database SHA-256 is unchanged from the pre-source baseline. One real source is now archived locally, and no labels or models were changed.

The user authorized publishing the repository to https://github.com/ombhanushaliii/pilot.git. Added origin and rewrote README.md to describe scope, current API sampling, manual source acquisition, unsuccessful Playwright attempts, progress and working rules concisely. Git authentication from this agent process failed, and Git Credential Manager returned access denied while enumerating its Windows credential store. The user's ordinary PowerShell Git session may still work. No credentials were read, exported or copied. Publication requires a successful Git push; no GitHub CLI login or plugin installation is required for that command.
