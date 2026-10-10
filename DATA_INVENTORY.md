# Verified contest-002 inventory

Verified against the saved dataset on 10 October 2026. The collection implementation is checkpointed in commit d7cc414. Generated data and reports remain local and ignored by Git.

## Population and development sample

Target: Codeforces Round 1124 (contest 2269). Earlier panel: 20 preceding standard CF Div. 2 rounds within 180 days, with varying durations recorded. Coverage refers to this panel, not lifetime account history.

| Coverage | Eligible target population | Development sample |
|---|---:|---:|
| At least five earlier panel rounds | 4,059 | 700 |
| One to four earlier panel rounds | 4,856 | 200 |
| None observed in the panel | 1,429 | 100 |
| Total | 10,344 | 1,000 |

The full target population is preserved. Sampling uses seed 736; per-group inclusion probabilities are recorded. Activity coverage is not a misconduct label.

## Submission records for the 1,000 sampled participants

- Official target submissions: 4,286.
- Eligible official submissions from the earlier panel: 30,232.
- Official target plus earlier submissions: 34,518.
- Additional retained target records from other participation/window categories: 675.
- Total stored submission records belonging to sampled handles: 35,193.

Each contains handle/author party, submission ID, contest ID, creation timestamp, relative contest time, problem metadata, programming language, verdict, testset, passed-test count, execution time and memory usage. Saved problem metadata includes index, name, type, maximum points, rating and tags. Execution time is program execution, not time spent writing code. Judging values are retrospective snapshots, not a history of when feedback appeared.

Target standings are available for 855 sampled participants. Saved standings include rank, points, penalty, per-problem results and hack counts. Missing standings and hack counts are not proof of misconduct and cannot be used as automatic labels. Earlier full standings were not collected in this run.

Contest metadata includes start time, duration, scoring type and phase. Coverage records include earlier participation IDs/counts, history group, selection flag and selection probability. Raw responses, timestamps, request parameters, checksums and pagination checkpoints preserve provenance.

## Entire run

- 2,306,696 distinct records traversed across 21 ID-bounded contest snapshots, including all participation categories.
- 285 successful raw responses archived.
- 107,666 normalized submissions: complete target snapshot plus eligible earlier submissions for the development sample.
- 21,121 individual-author target candidates, including excluded participation; all labels unknown.
- 33 automated tests passed at the implementation checkpoint. Final collection audit and actual zero-network rerun passed.
- Response-manifest SHA256: 64894a551ae2018f855b78965fef430c4526174b1af5d26a3f979935cc9e2e09.

## Missing before model training

No source code, code-change measurements, lifetime histories, per-user rating histories, profile demographics, evidence-backed labels, feature matrix or train/validation/test partitions have been built for contest-002. College and country are not collected for this run. None of the 1,000 participants has been assigned a fair/cheating class.

Dataset: data/pilot.sqlite. Raw responses: data/raw/contest-002/. Reports: reports/contest-002/. The response manifest identifies the local archives; a code commit does not embed their contents in Git.
