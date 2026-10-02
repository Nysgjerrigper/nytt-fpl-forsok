# Open work

Status: 2026-10-02. Recovery is submitted in
[PR #3](https://github.com/Nysgjerrigper/nytt-fpl-forsok/pull/3), whose GitHub state records integration.
Current verification and scoring controls are in [HANDOFF.md](HANDOFF.md).
Rules live only in [CLAUDE.md](CLAUDE.md), and historical audit IDs remain in the archive.

## Integration queue

1. Exercise the weekly path against deadline-current team state after integration,
   keeping prospective research isolated. Tests, package imports, cache equivalence
   and same-input MILP comparisons are complete; offline ingestion tests do not establish
   a new live end-to-end run.
2. Profile the optional [Open FPL adapter](docs/OPEN_FPL_ADAPTER.md) before default migration.
   Mapping, modern-rule checks and the first matched standard/origin comparison are
   complete. It was about 1.6x slower and both point CIs included zero. Preserve exact
   optimality; investigate formulation cost and evaluate realistic owner price/chip state.
   Price sensitivity and cross-position calibration remain research questions.
3. Review historical product-guide/output-path changes selectively. The August squad advice
   is dated evidence; preseason features need deadline/duplicate checks and registered evaluation.
4. After PO-authorized integration, preserve then retire the superseded source worktrees.
   Do not merge the five-commit validation branch wholesale.

## Research gates and parked ideas

- Neural studies: blocked on a proven runtime fix. Do not repeat known PyTabKit crashes.
- Stable MoE and MID per-player selection: rejected; no unchanged reruns.
- Bookmaker odds: parked pending historical data acquisition.
- Older history: needs position/team joins and evaluation; no change to start season now.
- Minutes cross-fitting: deprioritized after two hurdle comparisons failed to improve points.
- Conformal calibration: the c1d1 diagnostic was null; keep as historical research, not a
  promised improvement or production input.
- Probabilistic buckets/haul tilt: forecasting-only/rejected for decision use as documented.

Completed recovery work is in the dated handoff/research log; it is not repeated as an
open task. Git history purging is not proposed: it requires irreversible rewriting and
has no demonstrated benefit to this cleanup.
