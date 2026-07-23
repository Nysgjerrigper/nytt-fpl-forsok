# TODO

Open threads only. Completed items and their full rationale live in `RESEARCH_LOG.md`
(and `AUDIT_2026-07-11.md` for the A/B/C/D/E finding IDs); this file was compacted
2026-07-23 after every audit cluster item (1.1-1.7, 2.1-2.4, 3.1-3.6, 4.1, 4.2, 4.3, 4.5, 4.8) closed. Residual from 4.3:
the 33 MB file remains in HISTORY (~159 MB .git); purging it needs a history rewrite +
force-push - PO's call, only worth it if clone size ever becomes a real annoyance.

**Standing decision rules** (details in RESEARCH_LOG 2026-07-11): judge modeling changes on
the standard-protocol realized-points backtest vs **2057** (GW153-183, chips disabled; re-baselined 2026-07-23 after the element-code identity fix) with a
`fpl.milp.compare_backtests` CI (window CI width ~±140 - differences inside that are ties);
deployment claims quote the honesty ladder (~1500/31 GWs); GW191-221 is SPENT for selection.

## Open — evaluation & optimizer

- **[1.8][LOW][B5] Auto-subs + vice-captain activation in backtest scoring** — currently
  ignored, understating absolute realized points equally for all configs. Implement a simple
  auto-sub simulation in optimize.py's scoring block, or document the omission in the report.
- **[2.5][PARKED][C5] Bookmaker odds features** — strongest exogenous signal in the
  literature, but historical odds acquisition is a real data project.

## Open — engineering hygiene

- **[4.4][LOW][E4] Proper packaging** — `pyproject.toml` + `pip install -e .`, drop the
  `sys.path.insert` hack from every module.
- **[4.6][LOW][E6] Feature-frame cache** — parquet keyed on (dataset hash, feature version);
  every entrypoint currently rebuilds ~163k rows of features from scratch.
- **[4.7][LOW][E7] Docs currency** — README still says `legacy/` holds the R code (deleted);
  `report/main.typ` cites stale backtest numbers/windows; regenerate when convenient
  (`typst compile report/main.typ report/main.pdf`).

## Open — modeling ideas (revisit only with backtest capacity)

- **Minutes cross-fitted features** (2.1 v2 remainder): P(played)/E[min] as *features* into the
  production regressor. DEPRIORITIZED after the 3-class hurdle also tied (2053 vs 2057,
  RESEARCH_LOG 2026-07-23) - two hurdle variants in a row suggest the minutes signal is
  already carried by the nailedness features.
- **Probabilistic interval calibration** — [p10,p90] coverage 0.88-0.93 vs ideal 0.80; a
  conformal adjustment on a held-out slice would tighten it. Forecasting-only view.
- **History extension to 2016-17** — needs a `players_raw.csv` join for position/team and
  loses `xP`; only worth it if a backtest beats the standing baseline. Diminishing returns
  likely after the 2020-21 extension's win.
- **Per-player model selection pilot** — plan on branch `experimental/per-player-model-selection`;
  single-position pilot recommended before any commitment. Not started.

## Settled / parked elsewhere (do not reopen without new evidence)

- Combination strategy: `single:catboost` (config constant; bake-off re-checks each train run).
- FT banking policy: cap 2, not the site's 5 (RESEARCH_LOG 2026-07-20).
- Buckets: forecasting-only; captaincy haul-tilt rejected; loss-branch archived as tag.
- LambdaRank: clearly negative (branch `exp/lambdarank`). Current-GW xP: confirmed leak, unusable.
- Bayesian MDP manager: underperforms; branch `experimental/bayesian-mdp-manager`.
