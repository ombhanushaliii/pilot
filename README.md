# Codeforces post contest pilot

Milestone 1 collects completed contest submissions and earlier participant histories for an RL course project. It builds an auditable dataset and leaves all labels unknown. Model training follows the evidence-label and split audit.

Python 3.11 or newer is sufficient. Public collection uses the standard library and needs no API key or package installation.

## Commands in PowerShell

```powershell
Set-Location D:\pilot
python -m unittest discover -s tests -v
python -m cf_pilot collect --config config/pilot.json
python -m cf_pilot check --run-id pilot-001
python -m cf_pilot collect --config config/contest.json
python -m cf_pilot check --run-id contest-002
python -m cf_pilot status
```

Repeat the collect command to resume an interruption or reuse the frozen cache. A run ID fixes its configuration and initial responses. For a different cohort, contests, limits or acquisition date, create a new config with a new run ID. A failed or capped job must stay visibly incomplete until resolved.

## Pilot sampling

The active workflow is `config/contest.json`, run `contest-002`, selection policy `standard_div2_v1`. It selects the latest completed standard CF Div. 2 contest at least seven days old, and the 20 latest earlier matching rounds within 180 days. Combined Div. 1 + Div. 2 and explicitly unrated rounds are excluded. Durations may vary and are recorded; later timing comparisons must account for duration. A supplied `target_contest_id` selects a particular mature matching round. It freezes these IDs from the first archived contest list. Selection does not use suspected behaviour, institution, nationality, rating adjustments or current profile rating.

The collector downloads the complete available target submission list and its full public standings. It builds participants from submission authors, including those missing from standings, and preserves practice/virtual/out-of-competition records with explicit exclusions. Earlier contest pages are archived in full; participation facts count only eligible official submissions by the target population. Pagination continues to an empty page even if the API returns fewer than the requested page size. Each contest traversal is bounded by its first submission ID. If new practice submissions shift an offset page, the collector finds the previous boundary ID in that page, discards the repeated prefix and records the realignment. A missing anchor fails visibly rather than inferring continuity. Newer submissions added after traversal started are outside that ID-bounded snapshot.

Coverage is participation in this fixed panel: regular = at least 5, limited = 1-4, none_observed = 0. A reproducible sample of up to 1,000 uses 70/20/10 group weights, redistributing shortages. Everyone remains in the population report, and each group has a recorded inclusion probability. Earlier submissions are normalized from local raw archives only for the development sample. No per-user HTTP history requests are needed. `history_complete` in a contest run means complete acquisition of this panel, not all historical account activity.

Contest reports add `contest_panel.json`, `participant_coverage.csv` (every eligible target participant), and `development_sample.csv`. Sampling is by history coverage only; skill stratification and extra per-user rating history are later refinements. No-history cases are retained and do not imply new accounts. Handle changes may make historical participation harder to link; the first version does not claim complete identity resolution.

### Original organisation checkpoint

The user supplied `https://codeforces.com/ratings/organization/5276`. The parser reads only the main organisation rating table, excluding sidebar/global profiles. It records all discovered handles for provenance, then chooses 24 using random seed 736. Neither organisation membership nor current leaderboard rating is a model feature.

The pilot chooses three completed Div. 2 contests with the most shared rated participation among the sampled handles, restricted to contests seven to 365 days old. That improves engineering coverage but is not a representative sample for estimating misconduct prevalence. Explicit `contest_ids` can override selection.

Every sampled handle is queried for every selected contest, including handles absent from the final standings. Full available standings are archived. This is a targeted cohort pilot, not a contest-wide census. The later dataset needs broader contest submissions, comparison participants and evidence-backed case selection.

## Collection guarantees

- A SQLite rate gate serializes request starts at intervals of at least 2.1 seconds across clients sharing the database.
- HTTP status, JSON and API status are checked. Temporary failures have bounded retries. A denial is recorded, never bypassed.
- Raw successful response bytes are compressed and checksummed before checkpoints advance.
- Submission IDs are unique. Adjacent pages overlap by one record and are deduplicated.
- Interrupted collection resumes at durable page checkpoints.
- Caps, failed pages and conflicting judging snapshots are visible and fail the collection audit.
- The first normalized submission snapshot is retained; changed later judging fields are logged, not silently overwritten.
- Historical pagination stops on exhaustion or after two consecutive pages older than the lookback cutoff. Because the API sorts by submission ID, this lookback stop is a documented heuristic, not proof that every historical record exists.
- Eligible submissions are individual CONTESTANT entries in the half-open contest time window. Practice, virtual, team and ghost records are excluded. Out-of-competition records remain available for manual eligibility review.
- Prior contest context comes only from contests that finished before the target contest started. Appending future activity cannot change that earlier context.

The API can remove or alter data before it is first downloaded. This collector preserves what it received; it cannot restore records no longer available from the source.

## Generated data

`data/pilot.sqlite` stores jobs, requests, cohort provenance, contests, submissions, rating histories, standings, sessions and labels. `data/raw/<run_id>/` stores archived responses.

`reports/<run_id>/` contains:

- `data_audit.json`: collection status, counts, exclusions, coverage and conflicts.
- `sessions.csv`: participant-contest candidates, eligibility and prior-history counts.
- `response_manifest.json` and `.sha256`: archived response identities and checksums.
- `labels_for_review.csv`: unknown labels with fields for evidence and review notes. Existing worksheets are never overwritten.

Editing the label worksheet currently does not update SQLite. Validated label import is a subsequent milestone.

Exit code 0 means collection checks passed for this targeted run. Exit code 2 means a failure, incomplete pagination or integrity issue. Collection completion does not mean training readiness.

## Tests and next work

Offline tests cover sidebar exclusion, reproducible sampling, cache reuse, request spacing, bounded failures, page overlap, restart, checksum corruption, changing snapshots, eligibility, unknown labels and exclusion of future contest context. Temporary test databases are created and removed only within this project's work directory.

Next work: audit contest population coverage, establish evidence-backed session labels and explicit presumed negatives, repeat across multiple target contests, build numerical features, freeze temporal/user partitions, and then implement supervised baselines and the RL environment. Never convert unusual timing, SKIPPED, absent standings, college membership or rating reversals directly into cheating labels.
