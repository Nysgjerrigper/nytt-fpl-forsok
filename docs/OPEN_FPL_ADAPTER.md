# Open FPL Solver integration

The `solio` backend adapts our forecasts to pinned Open FPL Solver revision
`ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79`. The forecasting strategy remains CatBoost;
our corrected autosub/vice scorer remains the evaluation authority. The legacy optimizer
stays available as a historical reference while the candidate is evaluated.

## What is included

- Offline stable-code/global-week to element-ID/local-week mapping, with checked metadata
  and explicit currency conversion. Fixture predictions sum into player-round points.
- Exact forecast-origin admission and unique fixture identities. Repeated player-round
  rows need fixture or opponent/home identity; ambiguous duplicates fail. Historical and
  weekly forecast exports preserve these identifiers without changing the forecasts.
- Five-transfer banking and WC/FH bank preservation, separate chip inventories for each
  half, GW19 expiry, and no consecutive Free Hits or WC/FH in GW1.
- Original selling-price discounts until first sale; subsequent repurchases use market
  prices. FH preserves the permanent squad and bank.
- A legal XI and bench under Bench Boost; captain and vice remain distinct XI players.
- Exact HiGHS optimal status, valid solution and zero-gap requirements. Timeouts fail
  without exporting recommendations. An independent validator checks squad, formation,
  club limits, transfers, currency arithmetic and free-transfer transitions.

The adapted solve function is vendored locally, with preserved original source, LICENSE,
hash manifest and a reviewable patch. `scripts/vendor_open_fpl.py` regenerates it offline.
Upstream network ingestion, projection readers and its Gurobi command path are not used
by this adapter. Our wrapper always selects HiGHS.

The upstream package declares Python 3.14+. The isolated solve function is separately
tested on our Python 3.11 environment and pinned dependencies; this does not claim that
the whole upstream application supports Python 3.11. Upstream licensing includes Apache
2.0 text and separate commercial-use wording, both preserved in the bundled LICENSE.

## Use existing forecasts

```bash
python -m fpl.milp.solio --predictions-csv <predictions.csv> --season 2026-27 \
  --start-gw 232 --max-gw 234 --horizon 3 --initial-squad <stable-codes> \
  --initial-budget <bank-in-tenths> --initial-ft <available-transfers> \
  --state-json <state.json> --output <new-output.csv>
```

No fitting or network access occurs here. The given global weeks are examples for the
current 2020-21 start convention, not current deadline advice. Output paths must be new.
Omit `--initial-squad` for a fresh £100m draft. Fresh drafts do not describe an existing
team's transfer entitlement.

An example state document:

```json
{
  "sell_prices": {"123456": 52},
  "chip_inventory": {
    "wc": {"1": 1, "2": 1}, "fh": {"1": 0, "2": 1},
    "bb": {"1": 1, "2": 1}, "tc": {"1": 1, "2": 1}
  },
  "forced_chips": {"bb": [19]},
  "previous_free_hit_gw": 3
}
```

Replace IDs, prices and state with actual values. Chip counts are remaining inventories
per half; omitted chips are disabled. Forced targets and previous FH use **local** weeks.
Sell prices use stable player codes and tenths (£5.2m = 52). If sell prices are omitted,
buy-price accounting is explicit fallback behavior, not exact owner selling-price history.

The weekly driver accepts `--optimizer solio` and `--optimizer-state-json <state.json>`.
It supplies official bootstrap identities/prices and uses the same forecasts as legacy.
The weekly driver still refreshes/refits; never use it to construct a prospective holdout.
The backend default remains legacy until decision evidence supports switching.

## Limits and comparison protocol

Prices, clubs and positions must be static across an input horizon. Unknown owned players
without forecast metadata fail rather than disappearing. A blank player-round is zero
when player metadata exists. Forecast minutes are display inputs, never realized minutes.
Every requested round needs forecast coverage; a proven whole-round blank needs explicit
zero rows. The adapter cannot establish individual blank completeness from missing rows.
Bench Boost retains small bench-priority objective weights in addition to full boosted
bench points, matching the retained heuristic objective rather than pure expected points.

The objective initially retains our point, captain, vice and bench weights without new
cash/FT incentives. Both backends use HiGHS; faster execution is a measured hypothesis.
Modern FT rules differ from the legacy cap/reset policy, so differences cannot all be
attributed to solver implementation.

`actual_total_points` retains the historical gross scoring convention. The adapter also
exports `actual_net_points`, subtracting transfer hits. Compare both, and never mix gross
and net totals under one baseline label.

The registered first comparison uses preserved standard/origin forecasts, GW153-183,
horizon three, chips disabled and seed zero. Both receive the same completed player pool
and explicitly frozen first-available prices. This is a static-price engineering diagnostic,
not the previous variable-price control or a prospective promotion result. Source/input
hashes, commands, outputs and elapsed times are preserved in the dated experiment folder.

## Initial results

All 124 solves were optimal. Legacy versus Solio: standard 2144 versus 2126 points,
84.2 versus 133.7 seconds; origin 1902 versus 1921 points, 88.0 versus 141.4 seconds.
Paired Solio-minus-legacy 95% CIs were [-113,58] and [-64,82]. No hits were taken, so
gross and net agree. Solio was about 1.6x slower on this local workload; no general speed
claim or point improvement follows. Legacy remains the default.

Admission checks were tightened after the evaluation; all 62 prepared windows remain
byte/value equivalent and evaluated solver bytes are unchanged. Preserved evaluated
source and final-admission verification make that lineage explicit.
