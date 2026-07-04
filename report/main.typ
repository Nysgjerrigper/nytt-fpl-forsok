#set document(title: "FPL Points Forecasting and Squad Optimization: Project Status Report")
#set page(numbering: "1", margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 20pt, weight: "bold")[Fantasy Premier League Points Forecasting and Squad Optimization]
  #v(0.3cm)
  #text(size: 13pt)[Project status report]
  #v(0.5cm)
  #text(size: 11pt, style: "italic")[Generated 2026-07-04]
]

#v(1cm)

#align(center)[
  #box(width: 80%)[
    #set align(left)
    *Abstract.* This report documents a Fantasy Premier League (FPL) points-prediction and
    squad-optimization system: a per-position ensemble of machine-learning regressors feeding a
    mixed-integer linear program (MILP) that selects squads, transfers, captaincy and chips.
    Originally a Master's thesis (LSTM in R + MILP in Python), it has been rewritten as a single
    weekly-runnable Python pipeline and validated against the original system on an identical
    backtest: 1,966 realized points vs. the LSTM's 1,526 (+29%) over the same 31 gameweeks. All
    experiments - including the ones that failed, and one unresolved regression - are reported.
  ]
]

#v(0.8cm)

= Introduction

The system predicts FPL player points one to several gameweeks ahead and converts those forecasts
into weekly squad decisions via a MILP based on Kristiansen et al. (2018). The guiding principle
of the current phase is honest empirical validation: every candidate technique is tested against
the production system on a fixed backtest, and negative results are reported rather than
discarded. This report gives the data and methodology, the accumulated results, and the known
methodological limitations identified by an internal code and concept review.

= Data

Per-gameweek player statistics are sourced from Vaastav Anand's public mirror of the official FPL
API (`vaastav/Fantasy-Premier-League` on GitHub), covering six seasons - 2020-21 through 2025-26,
162,981 player-gameweek rows. The two oldest seasons lack the Opta expected-goals family and the
`starts` column (introduced 2022-23); these are treated as missing, which the tree-based models
handle natively. Gameweeks are indexed by a single ascending counter (`GW_global`): season $N$ in
the dataset occupies gameweeks $(N-1) times 38 + 1$ through $N times 38$, so the 2024-25 season's
gameweeks 1-31 - the validation window used throughout - is GW153-183.

Exploratory analysis (`notebooks/eda.ipynb`, all six seasons) establishes the two facts the
methodology is built around. First, `total_points` is heavily right-skewed and zero-inflated at
every position (Shapiro-Wilk rejects normality at $p < 0.001$ throughout), which motivates a
scale-free error metric (@sec-metrics) and models that do not assume Gaussian errors. Second, the
distribution is stable across seasons (per-position means for players who featured vary within
roughly 2.4-3.8 points across all six seasons, medians flat at 1-2), with one caveat: over six
seasons the average defender score-per-gameweek series no longer passes an Augmented
Dickey-Fuller stationarity test ($p = 0.41$; GK/MID/FWD all reject the unit root at 5%),
plausibly reflecting the 2025-26 `defensive_contribution` scoring change - a reason for caution
with fixed-mean time-series baselines at DEF specifically.

= Methodology

== Feature engineering <sec-features>

Each player-gameweek row carries roughly 80 engineered features:

- *Form features.* Rolling 3- and 5-gameweek means, previous-gameweek value, and a season-to-date
  expanding mean over ~18 per-gameweek statistics (points, minutes, xG family, BPS, ICT index,
  price, ownership, etc.). All are shifted one gameweek relative to the target row, so no row's
  features contain its own outcome.
- *Fixture-difficulty features.* The official FPL Fixture Difficulty Rating (FDR, 1-5) of the
  opponent faced that gameweek, and the mean FDR over this plus the next two scheduled fixtures.
  These are deliberately _not_ shifted: fixture lists are published before gameweeks are played,
  so they are known-ahead inputs, not leakage.
- *Minutes-projection features.* Rolling 5-game start rate and 60+-minute-appearance rate,
  shifted like the form features - a "will he actually play?" signal, since a benched player
  scores approximately zero regardless of ability.

== Forecasting models <sec-models>

Four independent models are trained, one per position (GK/DEF/MID/FWD): the statistics that
predict a goalkeeper's points are largely disjoint from a forward's, and pooling risks one
position's scale dominating the loss.

Per position, every regressor in a fixed registry is trained on the same features: LightGBM,
XGBoost, CatBoost, plain OLS, Ridge, ElasticNet, partial least squares (PLS), Random Forest,
Extra Trees, $k$-nearest neighbors, linear support-vector regression (LinearSVR), and a
sample-capped RBF-kernel SVR. Gradient-boosted models receive raw features including missing
values; linear, kernel and distance models are wrapped in a zero-imputer and standardizer.
Registry members that preliminary tests found unhelpful are retained rather than deleted - the
blending step (below) simply assigns them near-zero weight, and keeping them preserves the
comparison for later revisiting.

The registry's predictions are combined by non-negative least squares (NNLS): weights $w >= 0$
minimizing $norm(sum_i w_i hat(y)_i - y)$, normalized to sum to one. Two design choices matter
methodologically: (a) weights are fit on one half of the held-out test window and the ensemble is
evaluated on the _other_ half, so reported ensemble accuracy is a genuine holdout figure; (b)
plain OLS is designated the _index_ - the simple benchmark every technique must beat, in the
sense a passive market index is the bar an active strategy has to clear.

Alongside the point-forecast ensemble, per-position LightGBM quantile regressors (p10/p50/p90)
produce a prediction interval per player-gameweek. Quantile crossing is repaired by row-wise
sorting. This probabilistic view exists because two players with equal expected points are not
equal decisions - upside matters on the 2x captain multiplier - but it does not yet feed the
optimizer.

== Evaluation design <sec-eval>

Three complementary layers, in increasing order of decision-relevance:

+ *Static split.* Train on GW $<=$ 152, evaluate on GW153-183 - the same 2024-25 GW1-31 window
  the original LSTM was validated on, keeping every comparison in this report on one fixed
  window.
+ *Walk-forward validation.* For each gameweek from a start point, train on strictly earlier data
  and predict that gameweek only, rolling forward - many genuinely out-of-sample evaluations
  rather than one.
+ *Actual-points backtest (gold standard).* Walk-forward predictions are fed through the MILP
  over GW153-183, and the resulting squads are scored with _realized_ points. This is the check
  that matters: an accuracy gain that does not survive contact with the optimizer is not an
  improvement (@sec-fixmin shows exactly this happening).

== Error metrics <sec-metrics>

Point forecasts are scored with MAE and, primarily, MASE (Hyndman & Koehler, 2006): MAE divided
by the in-sample MAE of a naive "same as last gameweek" forecast, with the scale fit on training
data only. For a zero-inflated target, raw MAE is uninterpretable in isolation; MASE < 1 means
the model beats the naive floor regardless of the target's scale. Probabilistic forecasts are
scored with pinball loss (the proper scoring rule for quantiles) and the empirical coverage of
the [p10, p90] band against its nominal 0.80.

== Squad optimization <sec-milp>

The MILP follows Kristiansen et al. (2018): a 15-player squad (2 GK / 5 DEF / 5 MID / 3 FWD)
under a budget, at most 3 players per club, an 11-player lineup under formation constraints,
captain and vice-captain, a limited number of free transfers with a 4-point penalty per extra
transfer, and one-shot chip logic (two wildcards, free hit, bench boost, triple captain). It is
solved as a rolling horizon: re-solved each gameweek over a configurable lookahead, with only the
first gameweek's decision locked in. Venter & van Vuuren (2024) - whose formulation matches this
one almost exactly - attribute their strong case-study result (top 4.08% of ~8.24M managers) to
lookahead and forecast quality rather than optimizer sophistication, which is why this project's
effort concentrates on forecasting.

= Results

== Backtest lineage

All systems scored by realized points over the same 2024-25 GW1-31 window, identical optimizer:

#align(center)[
  #table(
    columns: 2,
    align: (left, right),
    table.header([*System*], [*Actual points*]),
    [Old LSTM + MILP (thesis)], [1526],
    [LightGBM + MILP, 4-season history], [1811],
    [Ensemble + MILP, 4-season history], [1900],
    [Ensemble + MILP, 6-season history], [*1966*],
    [+ fixture/minutes features], [1880 #footnote[Unresolved regression despite improved forecast accuracy - see @sec-fixmin.]],
  )
]

The best validated configuration (1,966) improves on the thesis system by 29%. Extending history
from four to six seasons was an unambiguous win: MASE fell at every position (FWD below 1.0 for
the first time in the project), and realized points rose from 1,900 to 1,966 (+3.5%).

== Forecasting-technique experiments

Simple and econometric baselines - rolling mean, naive drift, SES, Holt, Theta, Croston, pooled
AR(1), per-player ARIMA(1,0,1) - were each tested per position against the ensemble. None beats
it anywhere; SES is the best of the simple methods, Croston the worst (its intermittent-demand
design reacts too slowly when a player goes cold, and pooling it by position hides the per-player
heterogeneity the source literature exploited). Full tables in `RESEARCH_LOG.md`. Plain OLS - the
designated index - beats every one of these simple baselines, and the ensemble in turn beats OLS
at every position on four-season data; on six-season data OLS narrowly overtakes the ensemble at
GK only (MASE 0.579 vs 0.595), plausibly because goalkeeping is a small, simple-signal position
that gains more from data volume than from ensemble complexity.

A preliminary standalone check of the newest registry members (static split, current feature
set) found *LinearSVR beating the full ensemble at every position* (e.g. DEF 0.713 vs 0.799,
FWD 0.869 vs 0.959 MASE) - the largest single-model result so far, plausibly because its
epsilon-insensitive loss tracks the MAE-family metrics better than squared-error losses on a
zero-inflated target with outlier hauls. XGBoost and RBF SVR underperformed and are retained as
near-zero-weight blend candidates. Full-ensemble re-evaluation with the expanded registry was in
progress at the time of writing.

== Fixture/minutes features: better forecasts, worse squads <sec-fixmin>

Adding the fixture-difficulty and minutes-projection features improved MASE at every position
(GK 0.595→0.588, DEF 0.830→0.799, MID 0.811→0.799, FWD 0.984→0.959; DEF most, consistent with
clean-sheet dependence) - yet the actual-points backtest _fell_ from 1,966 to 1,880. The leading
hypothesis: the features smooth the mean forecast toward safe, nailed players, and the MILP -
which maximizes expected points and is blind to variance - loses the high-ceiling captaincy
differentials that drive realized hauls. The features are committed but explicitly not declared
a net win; resolving this divergence (noise check on a second window, or feeding the
probabilistic upside signal into captaincy) is the top open item. This is the project's clearest
demonstration of why the actual-points backtest, not MASE, is the gold-standard check.

The probabilistic module's [p10, p90] coverage is 0.88-0.93 against the nominal 0.80 - somewhat
over-wide intervals, expected where the p10 quantile pins at zero for blank-prone players.
Usable for relative risk ranking; conformal calibration is a possible refinement.

== Exploratory branches

Two isolated branches, neither merged: a Bayesian belief-state MDP manager after Matthews et al.
(2012) (1,083 points myopic / 847 Q-learning on the four-season window where the production
system scored 1,900 - expected, given necessary simplifications documented on the branch), and a
per-player model-selection plan (Venter & van Vuuren's per-player method choice; feasibility
capped by ~35% of live players having no prior-season history - recommendation: single-position
pilot before any commitment).

= Known methodological limitations <sec-limitations>

An internal review of the code and modeling concepts (2026-07-04) identified the following. None
invalidates the relative comparisons above; two mildly inflate absolute backtest numbers, and two
affect live use only. All are tracked in `TODO.md`.

+ *Blend-weight leakage into the backtest window.* Ensemble blend weights are fit on the first
  half of GW153-183 and then reused by the walk-forward backtest over that same window, so the
  first half's predictions use weights fit on their own outcomes. Absolute headline numbers
  (1,966 / 1,880) are modestly optimistic; the comparison between them is fair since both leak
  identically. Fix: fit backtest weights strictly before the window.
+ *Index-check asymmetry.* The ensemble's MASE is (correctly) measured on the test window's
  second half only, but the OLS index's on the full window - the "beats index" verdict compares
  slightly different evaluation sets.
+ *Live-mode feature staleness.* The weekly driver re-uses each player's last played row for
  future gameweeks, so (a) live fixture-difficulty features describe last week's fixture rather
  than the upcoming one, and (b) rolling form excludes the player's most recent match. Backtests
  are unaffected; live recommendations are degraded until fixed.
+ *Optimizer rule currency.* The MILP caps banked free transfers at 2 (the pre-2024-25 rule; now
  5) and assumes players sell at current market value (real FPL: purchase price plus half the
  profit). Fine for historical comparison, wrong for live 2026-27 play.

= Conclusion and future work

The rewritten pipeline beats its thesis-era predecessor by 29% in realized backtest points, and
the phase's central negative finding is itself useful: no simpler technique tested - econometric
or ML - beats the per-position ensemble, so further gains must come from architecture (per-player
selection, probabilistic captaincy) rather than from adding model types. Priorities, in order:
resolve the fixture/minutes MASE-vs-points divergence; fix the blend-weight leakage and live-mode
staleness from @sec-limitations; evaluate LinearSVR inside the full ensemble; then the per-player
selection pilot.

#pagebreak()

= References

#set par(justify: false, hanging-indent: 1cm)

Croston, J. D. (1972). Forecasting and stock control for intermittent demands. _Journal of the
Operational Research Society_, 23(3), 289-303.

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy.
_International Journal of Forecasting_, 22(4), 679-688.

Kristiansen, B. K., Gupta, A., & Eilertsen, W. (2018). _Developing a forecast-based optimisation
model for Fantasy Premier League_ (MSc thesis). Norwegian University of Science and Technology,
Trondheim.

Matthews, T., Ramchurn, S. D., & Chalkiadakis, G. (2012). Competing with humans at fantasy
football: Team formation in large partially-observable domains. _Proceedings of the Twenty-Sixth
AAAI Conference on Artificial Intelligence_, 1394-1400.

Venter, V., & van Vuuren, J. H. (2024). An optimisation approach towards soccer Fantasy Premiere
League team selection. _ORiON_, 40(1), 69-107.
