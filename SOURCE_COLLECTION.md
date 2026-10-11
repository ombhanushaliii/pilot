# Playwright source feasibility

Source acquisition is a separate optional workflow. It never modifies the public API SQLite database, feature matrices or reviewed labels. Source artifacts support later independent review, not automatic cheating classifications.

Current live status, 11 October 2026: source access is blocked. The user encountered a looping Cloudflare challenge in the Playwright login window. Stop that login attempt rather than repeating checkboxes. Cloudflare does not support Playwright for solving production challenges: https://developers.cloudflare.com/cloudflare-challenges/reference/supported-browsers/ . No source files or login-ready marker have been acquired. Ordinary viewing in the existing Helium session is the next manual feasibility check; it has not yet been verified. A small manually saved sample or an authorized data export is a fallback. No stealth or challenge bypass is implemented.

## Setup and sign-in

Python 3.11+ and the packages in requirements-sources.txt are needed. This machine already has Playwright 1.51.0, psutil 7.0.0 and native Edge/Chrome. No WSL is used. The default is Edge (`msedge`). If needed, install optional packages with `python -m pip install -r requirements-sources.txt`. The standard-library API collector remains unchanged.

From D:\pilot:

```powershell
python -m cf_pilot.sources login
python -m cf_pilot.sources collect --limit 5
```

The login command opens a visible, separate browser profile for manual sign-in. Detection of the Codeforces logout link closes the window and saves the profile. It waits up to ten minutes. No password is requested by the collector. Helium's existing profile is not read or copied. Authentication lives in ignored work/source-browser-profile/, never in source artifacts or Git. Treat that whole directory as secret account state. `--channel chrome` is another installed-browser option, but use the same channel for login and collection. An interrupted profile lock requires confirming the old process stopped before removing the lock file.

Collection is headless by default. `--headed` provides a visible diagnostic mode with the same profile, not a challenge bypass. Access denials, challenges, unexpected redirects, missing source, ambiguous table identity and browser errors stop the batch without retries. Manual login does not guarantee that automated source access will work. Frozen successful artifacts are retained, and every remaining task stays incomplete.

## Limited acquisition and integrity

config/source-feasibility.json specifies exact submission IDs and expected authors. The five-page feasibility sample uses three unverified user-supplied leads plus two ordinary sampled sessions, whose labels also remain unknown. The manifest is frozen at first collection under data/sources/source-feasibility-001/manifest.json. This mixed-format sample checks transport/extraction only; it is not a training or accuracy benchmark. A new manifest requires a new run ID.

One context and one reusable page process tasks sequentially, with a seven-second minimum interval between submission navigation starts within a batch. Initial document loading can make further subresource requests. This is our conservative website cadence, not the documented API rate limit. The first five entries are selected by default. `--limit 25` must be explicit for a larger manifest.

Extraction requires one #program-source-text element and an exact submission table row with a single matching profile handle, matching contest/problem and a nonempty language. The extractor preserves DOM textContent as UTF-8, including indentation and trailing line breaks. It does not claim to recover the original uploaded bytes or line-ending format. No source is executed. Page selectors and metadata mappings require live feasibility validation before scaling.

Files are <contest>-<submission>.source.txt and a JSON record of task, URL, author, problem, language, retrieval time, byte count, SHA-256 and unknown label. Source-only artifacts exclude account-page HTML, cookies, headers, screenshots, traces and HAR files. Cached records must match the frozen task and source checksum. Completed sources are never silently overwritten. An orphan source after an interrupted write fails visibly for manual recovery.

attempts.jsonl records outcomes with fixed failure reasons; last_result.json reports selected batch completion separately from whole-manifest completion. A completed cached rerun makes zero source navigations and does not launch a browser.

## Resources and testing

The collector samples itself and descendant processes every 0.5 seconds using psutil. It records the peak sum of working sets (RSS) and minimum available system memory. Shared pages can be counted more than once; this is not unique physical memory/PSS. Startup, navigation and cache-related peaks may be missed between samples. Unrelated browser processes are excluded. Defaults stop between navigation steps when available system memory falls below 1,024 MB or measured RSS sum exceeds 1,000 MB. These are guard thresholds, not a promise of steady memory usage or an instantaneous hard OS limit. The browser closes after stopping.

Run `python -m unittest discover -s tests -v`. The source tests check identity errors, denials, checksum corruption, frozen manifests, interrupted collection, cache-only reruns, cadence, selection limits, profile exclusion and memory stops. A separate local browser fixture check validates real DOM extraction. Live Codeforces extraction and whitespace must still be checked with actual pages. Independently verify any evidence claim before applying a label, and keep label-defining trap signatures out of behavioural predictors.

## Export from an already loaded Helium page

After the user reported that submission 368575830 opens normally in Helium, added a user-triggered bookmarklet and offline importer. No navigation, fetch, cookie access or browser launch occurs in the exporter. config/source-browser-export.json freezes the same five cases in a separate source-browser-export-001 run. Generated HTML and inspectable JavaScript are in the current Codex outputs directory. Build with python -m tools.build_source_exporter --output PATH.

Open source-exporter.html in Helium, drag Export CF pilot source into the bookmarks bar, open the first specified submission normally and click the bookmark when its source is visible. It downloads cf-source-2211-368575830.json. Save the file under D:\pilot\work and import with:

```powershell
Set-Location D:\pilot
& 'C:\Users\Om Bhanushali\AppData\Local\Programs\Python\Python313\python.exe' -m cf_pilot.source_exports 'D:\pilot\work\cf-source-2211-368575830.json'
```

The standard-library importer checks the export schema, UTC timestamp, canonical URL, frozen task, author, contest/problem, language and source uniqueness. It never executes code. Exact UTF-8 DOM text, SHA-256, source identity, export timestamp, import timestamp and export-file hash are retained. HTTP status is explicitly not observed for this route. The generic archive retrieval timestamp records archival time; exported_at_utc records the browser-reported capture time. Identical repeat imports are idempotent; conflicting or corrupt records fail without overwriting. Additional export fields are rejected, and the original JSON/HTML is not copied wholesale into the archive. Labels remain unknown.

64 tests passed, including a Node controlled-DOM fixture for the generated JavaScript and offline importer checks. Node syntax checking also passed. Live Helium extraction remains unverified until a real export is imported. No RAM benchmark or unattended source collection is claimed. The computer-use bridge failed to initialize, so no direct inspection of the user's Helium page was possible. If bookmarklets are blocked, retain manual export as an alternative without changing browser security settings.

## Current saved-page fallback

A real user-saved submission HTML page has now been imported successfully. Run:

```powershell
python -m cf_pilot.source_html "PATH-TO-SAVED-SUBMISSION.html" --submission-id 368575830
```

The importer checks the numeric # table column, author, contest/problem, language, page title and unique source against config/source-browser-export.json. It never runs HTML scripts. The original HTML stays outside the source archive; only extracted code, identity, SHA-256 and explicit provenance are stored. Capture time and HTTP status are unknown for this route. The saved page's Out of competition notice is a page fact, not a confirmed misconduct label or reconstruction of contest-time eligibility.

Version 2 of the bookmarklet also supports the plain numeric # column observed in the real saved page. Reopen work/source-exporter.html and drag Export CF pilot source v2 to the bookmarks bar to replace the previous version. Pretty-printed source is reconstructed from its line elements, with that mode recorded in metadata; it is not guaranteed to preserve the original uploaded file's whitespace. Unformatted DOM source retains exact textContent. Older JSON exports remain accepted by the importer.

The real HTML import and duplicate offline import succeeded; 67 tests passed after the correction. A successful live version-2 bookmarklet download is still pending. Do not interpret this extraction check as validation of the blog allegation.
