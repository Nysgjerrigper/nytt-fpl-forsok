# Origin capture-price replay

Registered before decisions in [issue #40](https://github.com/Nysgjerrigper/nytt-fpl-forsok/issues/40#issuecomment-6241380947).
Plan SHA256: `927ab9e2d1de49eaccc4144f2536e937901b5ab6c5b6d6e15ed7e27db2f76f0c`.

The preserved origin forecasts were replayed with capture-known prices held static
through each planning horizon. Both arms use the same registered-player pool,
modern Solio FT5 rules, no chips, fresh £100m state and acquisition-based
half-profit sales. This is an engineering diagnostic on already examined
GW141–144, not the actual historical rule policy or a production promotion study.

| Arm | Gross points | Net points | Native solve seconds | Explicit blank outcome rows |
| --- | ---: | ---: | ---: | ---: |
| Control | 197 | 197 | 7.95 | 4 |
| Candidate | 203 | 203 | 7.15 | 4 |

Candidate minus control is **+6 points**, paired block-bootstrap 95% CI
**[-8, +11]**, 10,000 resamples, block3, seed0. The comparator's raw report uses
control minus candidate: -6, CI [-11, +8]. Each arm wins two rounds. The interval
includes zero; no reliable superiority is established. Four rounds and one seed
provide very limited inference, and the original recipes change capacity as well
as loss. The comparison remains **UNVERIFIED**, unranked and ineligible for
promotion. Legacy-policy totals are separate and are not a clean price-only
sensitivity because backend/rules also differ.

Zero model fits; eight native solves, all returned Optimal with zero reported gap.
The pinned backend rejects nonoptimal native status. Full primal/dual native
certificates are not exported by this adapter, so snapshot native-certificate
assessment remains NOT ASSESSED. Each attempted solver input/state and returned
window solution was saved; stable-ID decisions preceded separate strict settlement.
All selected outcomes are covered, including four explicit retrospective blank
rows per arm. No old-policy reference totals were used for reconciliation.

At origin141 there are 2,081 admitted horizon rows; later origins admit
2,091/1,182/838. Origins142–144 each exclude one forecast row for a player absent
from that capture's register. Both arms have identical admitted fixture identities.
No future-row metadata/prices, outcome columns or default50 prices enter the
solver. No wholly absent player is added. The inherited Solio converter fills
missing player-round forecast cells with zero after the input boundary. This
planning assumption is not independently admitted blank evidence and was not an
explicit separate field in the preregistered plan; it remains a qualification gap. Capture metadata/buy prices, selling basis and prepared
frame hashes are retained in each decision receipt.

| Evidence gate | Assessment |
| --- | --- |
| Preserved origin forecast bytes | Hash-bound; no refits |
| Capture-price identity and recent predeadline timestamp | PASS under supplied manifests |
| Future raw price/metadata/outcome perturbation | Prepared-input invariance regression PASS |
| Solver outcome columns | ABSENT |
| Separate strict settlement and selected outcome coverage | PASS |
| Exact-deadline price / archive authenticity | NOT ASSESSED |
| Missing player-round forecast / blank planning validity | NOT ASSESSED; inherited zero-fill |
| Fixture-information causality | NOT ASSESSED |
| Complete outcome-blind execution | NOT ASSESSED |
| Full independent modern site-rule/financial legality | NOT ASSESSED |
| Promotion / default backend change | None |

Full immutable inputs and outputs remain in the task artifact folders
`OriginPriceReplayInputs_20261017` and `TrainingGroundOriginPriceReplay_20261017/first`.
The copied `complete.json` indexes that full external bundle, not just these tracked
summary files. Source/runtime/script and 17 input-file hashes were checked before
and after execution; all output hashes and both strict settlement receipts were
independently rechecked. Preserve those bundles when reproducing the script with
the registered absolute paths, or register a new plan for moved inputs.
