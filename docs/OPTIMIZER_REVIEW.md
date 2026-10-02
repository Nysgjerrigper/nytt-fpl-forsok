# Alternative optimizer review

Checked: 2026-10-02. This is a source review, not a completed replacement benchmark.
No external optimizer was installed or executed.

## Recommendation

Evaluate [solioanalytics/open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver)
as a separate adapter candidate using our existing forecasts. Pin the reviewed revision
[ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79](https://github.com/solioanalytics/open-fpl-solver/commit/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79),
dated 2026-09-15. It has an Apache 2.0 license and a multi-period solver with support
for five banked free transfers, bank preservation through Wildcard/Free Hit, and
existing-holding selling prices. It accepts custom forecast CSVs.

The reviewed code does **not** automatically enforce separate chip inventories for
each half of the season. Its chip limits apply to the supplied horizon, with optional
allowed/forced weeks. We must supply and test explicit half-season inventory and expiry
handling before considering it ready for 2026/27.

## Evidence and current-season requirements

The [official 2026/27 announcement](https://www.premierleague.com/en/news/4679873)
specifies five banked transfers and two sets of Wildcard, Free Hit, Triple Captain and
Bench Boost. The first set expires at the GW19 deadline. There is no AFCON transfer
top-up this season.

| Area | Reviewed source | Finding |
| --- | --- | --- |
| Transfer bank | [solver.py lines 86–104 and 482–511](https://github.com/solioanalytics/open-fpl-solver/blob/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79/dev/solver.py#L86-L104) | Five-transfer banking; WC/FH preservation is present. |
| Chips | [solver.py lines 522–541](https://github.com/solioanalytics/open-fpl-solver/blob/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79/dev/solver.py#L522-L541) | Horizon limits, without automatic per-half inventories. |
| Selling prices | [solver.py lines 48–80](https://github.com/solioanalytics/open-fpl-solver/blob/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79/dev/solver.py#L48-L80) | Half-profit selling logic for existing holdings; future prices remain static. |
| Forecast input | [data_parser.py lines 13–51](https://github.com/solioanalytics/open-fpl-solver/blob/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79/dev/data_parser.py#L13-L51) | Custom CSV needs current element ID, position, and local-week points/minutes. |
| Tests | [tests directory](https://github.com/solioanalytics/open-fpl-solver/tree/ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79/tests) | Reviewed tests cover option parsing, not optimizer rule behavior. |

Our forecasts use stable player codes and global gameweeks. An adapter must map these
to current element IDs and season-local weeks, reject missing or duplicate identities,
and preserve the forecast bytes used for comparison.

## Other candidates

[Linus-J/FPL-decision-engine](https://github.com/Linus-J/FPL-decision-engine) and
[MattBryantt/fpl](https://github.com/MattBryantt/fpl) advertise modern-season functionality.
Their complete rule implementations and licensing were not verified in this pass;
README claims do not establish replacement readiness. The older LaptopHeaven fork is
not the current maintained source; the Sertalp predecessor redirects to the Solio repo.

## Evaluation plan

1. Complete a pinned-source and license review, including half-season chip handling.
2. Build an isolated forecast/team-state adapter and independent rule tests: transfer
   banking, chip preservation and expiry, selling prices, FH reversion, and budget.
3. Compare both optimizers on identical predictions and team state. First establish
   legal decisions and reproducibility; then compare realized points and paired CIs.
4. Preserve the historical legacy-policy controls separately from modern-rule results.
5. Adopt only after review of rule fidelity and decision evidence. The current production
   forecast strategy remains `single:catboost`.
