# External fixture data admission

External records enter research through `fpl.data.external`. This module validates
byte identity, exact player mappings and chronological consistency. A successful
call is not independent proof that an upstream statistic was publicly available.
Source publication evidence must be audited before fitting or feature selection.

`load_external_records(csv_path, manifest_path, max_available_at=...)` hashes the
complete CSV and parses those same bytes. Its manifest requires schema
`external_fixture_v1`, a source, a 40-character immutable revision, a SHA256,
`snapshot_available_at`, a `timing_basis` of `archived_capture` or
`provider_publication`, and an HTTPS `publication_evidence` link. The registered
historical cap must follow the snapshot. Match dates and Git commit dates are
insufficient publication evidence and are refused as timing bases.

Rows require `season`, opaque string `external_player_id` and `event_id`,
`event_time` and `available_at`. Times need explicit ISO timezone offsets and are
normalized to UTC. Event time must not follow availability, which must not follow
the snapshot. Duplicate identity/version rows are refused; successive published
versions are permitted. Leading zeroes in source IDs and missing statistics survive
admission. Missing collection must never become zero activity.

`map_external_players` uses exact season-scoped IDs: external ID -> FPL element
ID -> stable FPL player code. Supply mapping columns `season`,
`external_player_id`, `fpl_element_id` and roster columns `season`,
`fpl_element_id`, `player_id`, `position`. Both mappings must be one-to-one within
a season and positions must be GK, DEF, MID or FWD. Names and fuzzy matching are
not used. Unmatched IDs fail by default; an explicit `allow_unmapped=True` retains
unknown identities with `mapping_status`. Such rows cannot silently enter fitting.
Mapping and roster publication times advance each record's effective availability;
they cannot backdate a later identity dictionary.

`latest_available_records(records, deadline, excluded_event_ids=...)` keeps only
events and their published versions strictly before the deadline, removes current
fixture IDs, and selects each event's latest available revision. The caller must
provide actual archived deadlines or a separately justified registered boundary.
This adapter does not infer deadlines from kickoff or season-ordinal gameweeks.
Validate cup tournament coverage, club membership at the event, and Elo's effective
time separately before constructing rest, workload or strength features.

## Historical source findings

The 2026-10-03 audit uses immutable repository revisions and historical samples.
Compact audit evidence is stored under `experiments/external_source_audit_2026-10-03/`;
research consequences and issue references are recorded in `RESEARCH_LOG.md`.

Vaastav has individual Understat histories from 2021/22 onward and season-scoped
ID dictionaries. A 2021/22 sample dictionary contains 609 unique Understat/FPL
ID pairs, all joining the 737-player FPL roster, whose stable codes are unique.
This establishes the sampled mapping's consistency, not full season coverage or
availability of that final dictionary at earlier deadlines. A player history can
contain multiple seasons and leagues; filter the admitted competition and season
before joining. The 2020/21 folder has team files and a 524-player season aggregate,
which cannot reconstruct player-match windows. Examined histories contain event
dates but no publication/capture timestamps. Immutable earlier blobs help recover
versions; Git timestamps alone do not prove original publication timing.

[FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) begins in 2024/25,
after our discovery and selection windows. Its roster exposes stable FPL codes,
but also staff/unknown positions. Gameweek folders are event partitions; the
exporter rewrites dynamic statistics after rounds finish. Folder names therefore
do not establish deadline availability. Cup/European coverage is uneven, and the
examined historical headers lack capture/publication fields. No records from this
source have been admitted to training or used to select new features.

[OpenFPL](https://github.com/daniegr/OpenFPL) primarily supplies pretrained models
and inference. Its paper's future-season imputation of missing older observations
is incompatible with this repository's temporal research protocol. Reusing feature
ideas requires fold-local preprocessing and independently admitted historical data.

Mapping preserves each revision's original `source_available_at`. Later mapping or
roster snapshots may give several revisions the same effective availability; their
original publication times still determine the latest revision after that mapping
becomes available. Ambiguous duplicate original revisions are refused.

## Historical deadline availability archive

The 2026-10-03 audit preserves captures from
[Randdalf/fplcache](https://github.com/Randdalf/fplcache/tree/de78720734e8c24954433fbf91bc356188cfc669)
at revision `de78720734e8c24954433fbf91bc356188cfc669` and
[unsaltedpw/fpl-armband](https://github.com/unsaltedpw/fpl-armband/tree/4cbbe8b206686a6def6751966c640fbc673333a7)
at revision `4cbbe8b206686a6def6751966c640fbc673333a7`. Repository filenames and
Git clocks do not prove historical publication. The admission evidence instead
binds each official FPL bootstrap response to an independently retrieved Internet
Archive CDX timestamp and SHA1 of its original response bytes (compressed when
served compressed), then checks the decoded body SHA256 against the preserved
source manifest. Fplcache's timezone inconsistencies and differing prices close
to deadlines make its samples corroboration only. Neither final deadline prices
nor complete player registration throughout the forecast horizon are proved.

`fpl.data.deadline_snapshots.load_deadline_rosters` verifies hashes before parsing,
stable player codes, source and payload identities, and capture/deadline clocks.
An admitted capture must be strictly before the deadline, at most 24 hours old,
and show the exact requested event as `is_next`. Invalid evidence fails; missing,
stale or earlier-event captures are explicitly unavailable. Validated optional
manager elements are excluded and counted, including 20 per origin from 2024/25
GW23 onward. Captured player availability is legitimate current external
information: it needs no outcome shift. The eight research features are six status
indicators (`a/d/i/s/u/n`), published next-round chance divided by 100, and capture
age in hours. Null chance remains NaN; missing archive coverage leaves all eight
features NaN. Target statistics are never copied from the archive. An origin
snapshot callback must freeze these same availability features across its full
forecast horizon, rather than read later target-round captures.

The saved admission audit records the following coverage. These counts describe
archive admission, not model quality or optimizer gains.

| Window | Independently verified captures | Admitted origins | Excluded origins |
|---|---:|---:|---|
| Discovery GW1–136 | 96 | 78/136 | 18 older than 24 hours; 40 without a verified capture |
| Selection GW137–152 | 16 | 16/16 | None |
| Historical reference GW153–183 | 31 | 28/31 | GW160/161/162: ages 225.849/377.349/562.849 hours (rounded) |

All 47 selection/reference capture proofs were independently verified. The
reference remains a repeatedly examined historical window; archive timing alone
does not make it an untouched holdout. No modeling improvement is established by
these checks.

Compact reports and audit scripts live under
`experiments/deadline_snapshot_audit_2026-10-03/`. Repeat admission offline with:

```bash
python experiments/deadline_snapshot_audit_2026-10-03/verify_admission.py \
  --output experiments/deadline_snapshot_audit_2026-10-03/admission_verification_current.json
```

The verifier reconstructs every origin through the production admission adapter,
compares counts, reasons, exact ages and manager exclusions with
`admission_results.json`, and writes `admission_verification.json` with input and
current source hashes. It does not fetch evidence, fit models or read outcomes.
Preserve any report already hash-bound by a completed study; current verification
uses a separate report because later pipeline edits can change its source hashes.
The actual raw corpus and indexes are ignored; local preserved copies are needed
for replication, including compressed source payloads, original Archive responses
and manifests. Upstream fpl-armband has no LICENSE at the pinned revision; its code
stays ignored and unvendored. No upstream source code is copied into the pipeline.

## Understat historical archive pilot (2026-10-04)

A bounded follow-up found independent Archive CDX entries for sampled Liverpool
team pages in 2020 and 2022. The original replay at
[2020-11-01 04:18:33 UTC](https://web.archive.org/web/20201101041833id_/https://understat.com/team/Liverpool/2020)
was preserved locally. Its decoded SHA1 matches CDX digest
`ROLN4L5ZQQGBKLVTNMPY5C23R7GAHZUQ`; decoded SHA256 is
`c185d7a7c599af3bf7d9c2d6ea7caf1af757b66ef730349eaeceba8626aba7da`.
Here the digest matches decoded HTML, not the replay's gzip transfer bytes.
Both byte forms and response headers are preserved separately. This result
establishes a recoverable, independently timed payload for one team/season.

The team page embeds 38 fixture records and 21 player season-to-capture
aggregates. Upcoming fixtures are distinct from completed results. Player
aggregates are not player-match histories and cannot reconstruct the intended
player rolling windows. Match timestamps have no explicit timezone, and URL
season scope is not an independently validated competition/season identity.
Those joins must be resolved before records enter the external adapter.

Monthly-collapsed, one-team metadata samples do not establish complete deadline
coverage. The 2021/2023 sampled queries returned transport failures; their
coverage remains unknown. The next bounded acquisition should inventory season,
team and match captures, measure identity/selection-window coverage and admit
only observations actually present in a verified capture before the forecast
deadline. Stale captures may supply older information but cannot fill intervening
unobserved matches. No Understat observation has entered model training.

Evidence and collector: `experiments/understat_archive_audit_2026-10-04/`.
Raw response files remain ignored local evidence; compact hashes are retained.

The preserved replay/index pair subsequently passed static extraction: seven
completed matches and 31 upcoming fixtures are explicitly distinguished. The
parser accepts only named static JSON assignments and uses a restricted escape
decoder, with no JavaScript execution. Duplicate assignments, duplicate JSON
keys/IDs, malformed/nonfinite values and oversized inputs fail closed. Source
numeric strings and event datetimes remain unchanged; extraction does not admit
records for training. See `parse_verification.json` and the synthetic parser tests.
