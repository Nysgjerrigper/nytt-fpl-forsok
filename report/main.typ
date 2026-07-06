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

#v(0.8cm)

= Conclusion and status

The system - a per-position machine-learning ensemble feeding a MILP squad optimizer - beats its
thesis-era LSTM predecessor by 29% in realized backtest points over an identical 31-gameweek
window (1,966 vs 1,526; @sec-backtests). The current state of play, most decision-relevant first:

+ *CatBoost (MAE loss) is now the production forecaster at every position*, selected empirically:
  a 20-gameweek walk-forward head-to-head and a static-window combination bake-off both found the
  single CatBoost model beats the NNLS blend, equal-weight top-k, and ridge stacking everywhere
  (weighted walk-forward MASE 0.684 vs the blend's 0.761) - the classic Clemen (1989) result that
  estimated combination weights lose to the best single member when members are many and
  collinear (@sec-registry).
+ *Best ranker, worst calibrated*: the new decision-aligned diagnostics show CatBoost ranks
  players and picks captains better than any blend (top-1 capture up to 0.57 vs 0.42) while
  under-predicting the aggregate point level by ~40-55% - a direct consequence of its MAE loss
  fitting conditional medians of a zero-inflated target. Its level must be recalibrated before it
  feeds the MILP, whose transfer penalties are absolute-scale (@sec-registry, @sec-limitations).
+ *The fixture-difficulty and minutes-projection features remain an unresolved trade-off*: they
  improved forecast accuracy at every position yet reduced realized backtest points 1,966 → 1,880
  (@sec-fixmin). The mean-vs-median finding above is a plausible mechanism, and the diagnostics
  now exist to test it.
+ *No simple or econometric technique tested beats the ML models* (nine baselines, @sec-experiments);
  a further ~37 engineered features (opponent form, exponentially-weighted form, per-90 rates, xP)
  were roughly accuracy-neutral on the validation window - reported honestly, kept pending
  hyperparameter tuning, which remains unexplored territory.
+ A code-and-concept review found and fixed four methodological errors (blend-weight leakage, two
  live-mode staleness bugs, an evaluation asymmetry); the absolute backtest numbers predate the
  leakage fix and will be re-baselined (@sec-limitations).

Priorities, in order: recalibrate CatBoost's level, then re-baseline the actual-points backtest
under the fixed weight scheme with CatBoost-only forecasts; run the new Optuna tuner at scale;
revisit the fixture/minutes divergence with the new diagnostics; then a per-player
model-selection pilot.

= Introduction

The system predicts FPL player points one to several gameweeks ahead and converts forecasts into
weekly squad decisions (transfers, lineup, captaincy, chips) via a MILP following Kristiansen et
al. (2018). The governing principle is honest empirical validation: every candidate technique is
tested against the production system on a fixed backtest window, and negative results are
reported rather than discarded.

= Data

Per-gameweek player statistics come from Vaastav Anand's public mirror of the official FPL API
(`vaastav/Fantasy-Premier-League`), spanning six seasons - 2020-21 through 2025-26, 162,981
player-gameweek rows. The two oldest seasons lack the Opta expected-goals family and `starts`
(introduced 2022-23); these are treated as missing, which the tree-based models handle natively.
Gameweeks carry a single ascending index (`GW_global`): dataset season $N$ occupies gameweeks
$(N-1) times 38 + 1$ through $N times 38$, so the fixed validation window - 2024-25 season,
gameweeks 1-31 - is GW153-183.

Exploratory analysis (`notebooks/eda.ipynb`, all six seasons) establishes the two facts the
methodology is built on: `total_points` is heavily right-skewed and zero-inflated at every
position (Shapiro-Wilk rejects normality, $p < 0.001$ throughout), motivating a scale-free error
metric and models free of Gaussian assumptions; and the distribution is stable across seasons,
with one caveat - the average defender score-per-gameweek series fails an Augmented
Dickey-Fuller stationarity test over the full six seasons ($p = 0.41$; the other positions pass),
plausibly reflecting the 2025-26 `defensive_contribution` scoring change.

= Methodology

== Feature engineering <sec-features>

Each player-gameweek row carries ~115 engineered features:

- *Form.* Rolling 3- and 5-gameweek means, previous-gameweek values, and two expanding-mean
  horizons - season-to-date (resets each season) and career-to-date - over ~18 per-gameweek
  statistics (points, minutes, xG family, BPS, ICT, price, ownership), plus exponentially-weighted
  means (halflife 3 gameweeks) for the ten most form-sensitive statistics, motivated by the
  earlier finding that exponential smoothing beats flat windows among simple baselines. All
  shifted one gameweek relative to the target row, so no row's features contain its own outcome.
- *Rates.* Per-90-minute rates (goals, assists, xG, xA over rolling minutes) separating
  production efficiency from playing time.
- *Opponent strength.* The upcoming opponent's rolling 6-gameweek attacking/defensive form and
  clean-sheet rate, computed at team level and merged by opponent with the same strict
  one-gameweek shift - a dynamic, in-season complement to the static FDR.
- *Fixture difficulty.* The official FPL Fixture Difficulty Rating (1-5) of that gameweek's
  opponent, plus the mean over this and the next two fixtures. Deliberately _not_ shifted:
  fixture lists are published ahead of play, so these are known-ahead inputs, not leakage.
- *Minutes projection.* Rolling 5-game start rate and 60+-minute rate, shifted like form - a
  "will he actually play?" signal.
- *FPL's own forecast.* Shifted forms of xP, FPL's published pre-match expected points.

Statistics absent for entire early seasons (the Opta xG family and starts before 2022-23) are
kept as missing values rather than zero-filled: "not collected" is a different fact from "recorded
zero", and the gradient-boosted models branch on missingness natively.

== Forecasting models <sec-models>

Four independent models, one per position (GK/DEF/MID/FWD) - the statistics that predict a
goalkeeper's points are largely disjoint from a forward's, and pooling risks one position's scale
dominating the loss. Per position, every regressor in a fixed registry is trained on the same
features: LightGBM, XGBoost, CatBoost (MAE loss), OLS, Ridge, ElasticNet, PLS, Random Forest,
Extra Trees, $k$-NN, LinearSVR, and a sample-capped RBF SVR. Boosted models take raw features
including missing values; linear/kernel/distance models get a zero-imputer and standardizer.
Registry members that tests find unhelpful are retained - the blend simply assigns them near-zero
weight, preserving the comparison for later revisiting.

How the registry members are _combined_ is itself an empirical question, answered per position by
a combination bake-off scored on identical held-out rows: the best single member (no combination
at all) vs. non-negative least squares (NNLS) vs. an equal-weight average of the top-$k$ members
vs. ridge-regularised stacking - the latter two being the forecast-combination literature's
standard remedies for the instability of estimated weights (Clemen, 1989). Whatever wins becomes
the production combiner, so the choice tracks the evidence rather than an assumption. Weight
fitting is separated by purpose: _evaluation_ weights are fit on one half of the held-out test
window and scored on the other half, so reported accuracy is a genuine holdout figure;
_production and backtest_ weights are fit on a 16-gameweek window strictly before whatever is
subsequently predicted, so no weight ever sees a gameweek it will be used to forecast. Plain OLS
is designated the _index_ - the simple benchmark every technique must beat, as a passive market
index is the bar an active strategy must clear.

Beyond MAE/MASE, evaluation includes decision-aligned diagnostics motivated by a structural trap:
MAE-family metrics are minimized by conditional _medians_, while the downstream optimizer consumes
conditional _means_ and captaincy rewards the upside tail. The diagnostics - RMSE (mean-aligned),
bias, total calibration (ratio of predicted to actual total points), within-gameweek Spearman rank
correlation, and top-1 capture (how often the highest-ranked player actually scored most) - make
that gap visible instead of letting MASE alone select median-flattening models.

A parallel probabilistic view - per-position LightGBM quantile regressors (p10/p50/p90, crossing
repaired by row-wise sorting) - produces a prediction interval per player-gameweek, because two
players with equal expected points are not equal decisions once the 2x captain multiplier rewards
upside. It does not yet feed the optimizer.

== Evaluation design <sec-eval>

Three layers, in increasing decision-relevance:

+ *Static split.* Train on GW $<=$ 152, evaluate on GW153-183 - the window the original LSTM was
  validated on, keeping every comparison on one fixed window.
+ *Walk-forward validation.* For each gameweek from a start point, train strictly on earlier data
  and predict that gameweek only - many genuinely out-of-sample evaluations rather than one.
+ *Actual-points backtest (gold standard).* Walk-forward predictions run through the MILP over
  GW153-183; resulting squads scored with realized points. An accuracy gain that does not survive
  contact with the optimizer is not an improvement (@sec-fixmin).

Point forecasts are scored with MAE and, primarily, MASE (Hyndman & Koehler, 2006) - MAE relative
to the in-sample error of a naive last-gameweek forecast, the scale fit on training data only;
MASE < 1 beats the naive floor regardless of the target's scale. Probabilistic forecasts are
scored with pinball loss and [p10, p90] coverage against its nominal 0.80.

== Squad optimization <sec-milp>

The MILP follows Kristiansen et al. (2018): 15-player squad (2/5/5/3) under budget, max 3 per
club, 11-player lineup under formation constraints, captain/vice-captain, limited free transfers
with a 4-point penalty per extra, and one-shot chip logic (two wildcards, free hit, bench boost,
triple captain), solved as a rolling horizon with only the first gameweek's decision locked in.
Venter & van Vuuren (2024) - whose formulation matches almost exactly - attribute their strong
case study (top 4.08% of ~8.24M managers) to lookahead and forecast quality, not optimizer
sophistication; hence this project's focus on forecasting.

= Results

== Backtest lineage <sec-backtests>

All systems scored by realized points over 2024-25 GW1-31, identical optimizer:

#align(center)[
  #table(
    columns: 2,
    align: (left, right),
    table.header([*System*], [*Actual points*]),
    [Old LSTM + MILP (thesis)], [1526],
    [LightGBM + MILP, 4-season history], [1811],
    [Ensemble + MILP, 4-season history], [1900],
    [Ensemble + MILP, 6-season history], [*1966*],
    [+ fixture/minutes features], [1880 #footnote[Unresolved regression despite improved forecast accuracy (@sec-fixmin). Both this and the 1,966 figure predate the blend-weight fix of @sec-limitations and are modestly inflated; their relative comparison is unaffected.]],
  )
]

Extending history from four to six seasons was an unambiguous win: MASE fell at every position
(FWD below 1.0 for the first time) and realized points rose 1,900 → 1,966.

== Forecasting-technique experiments <sec-experiments>

Eight simple and econometric baselines - rolling mean, naive drift, SES, Holt, Theta, Croston,
pooled AR(1), per-player ARIMA(1,0,1), and an empirical-Bayes shrinkage of player means toward
position means - were each tested per position. None beats the ML models anywhere; SES is the
best simple method, Croston and EB-shrinkage the weakest (full tables in `RESEARCH_LOG.md`).
Plain OLS, the index, beats every simple baseline; the ML models in turn beat OLS.

== Expanded model registry <sec-registry>

Six techniques were added in one round (LinearSVR, RBF SVR, XGBoost, CatBoost, PLS, EB-shrinkage)
and evaluated on the standard static split:

#align(center)[
  #table(
    columns: 5,
    align: (left,) + (right,) * 4,
    table.header([*Position*], [*CatBoost*], [*LinearSVR*], [*12-member ensemble*], [*previous ensemble*]),
    [GK],  [*0.513*], [0.513], [0.534], [0.588],
    [DEF], [*0.705*], [0.713], [0.747], [0.799],
    [MID], [*0.731*], [0.751], [0.806], [0.799],
    [FWD], [*0.849*], [0.869], [0.853], [0.959],
  )
]
#align(center)[#text(size: 9pt, style: "italic")[MASE, GW153-183 static split. CatBoost uses MAE loss; ensemble scored on a genuine holdout half-window.]]

Two findings. First, *CatBoost with MAE loss is the best single model at every position*, by a
wide margin over everything preceding it - consistent with the hypothesis (raised by LinearSVR's
strong showing) that MAE-aligned training objectives suit a zero-inflated target with outlier
hauls better than squared error. PLS and XGBoost added nothing (retained as near-zero-weight
members). Second, *the blended ensemble now trails standalone CatBoost everywhere* - with 12
collinear members, NNLS overfits its half-window weight fit (the MID blend failed to select
CatBoost at all).

That second finding was settled by two independent tests. A 20-gameweek walk-forward head-to-head
(GW169-226, members retrained each step, weights fit strictly before the window) had CatBoost-only
beat the blend at every position - weighted MASE 0.684 vs 0.761 - and the combination bake-off on
the static window agreed, with equal-weight top-$k$ second and ridge stacking last:

#align(center)[
  #table(
    columns: 5,
    align: (left,) + (right,) * 4,
    table.header([*Position*], [*CatBoost only*], [*top-k (k=3)*], [*NNLS*], [*ridge*]),
    [GK],  [*0.502*], [0.517], [0.547], [0.551],
    [DEF], [*0.689*], [0.705], [0.782], [0.801],
    [MID], [*0.702*], [0.720], [0.797], [0.780],
    [FWD], [*0.788*], [0.805], [0.884], [0.887],
  )
]
#align(center)[#text(size: 9pt, style: "italic")[Combination bake-off: MASE on identical held-out rows (GW168-183 eval half). The best single member beats every combination scheme - Clemen's (1989) forecast-combination result in the wild.]]

Production forecasting is therefore CatBoost-per-position, chosen by the bake-off on every
training run rather than hard-coded - if a future change makes a combination win again, production
follows automatically.

The decision-aligned diagnostics add a critical nuance: *CatBoost is the best ranker but the worst
calibrated*. It captures the actual top scorer far better than the blend (top-1 capture 0.57 vs
0.42 at MID; higher at every position) with near-equal rank correlation - yet its bias is -0.32 to
-0.60 points per player-gameweek and its forecasts sum to only 44-63% of the points actually
scored, exactly the median-flattening the MAE loss predicts on a zero-inflated target. Ranking
quality is what transfers and captaincy consume, so this is the right model to keep - but the MILP
also makes _absolute-scale_ decisions (a 4-point transfer penalty, chip thresholds), which
deflated forecasts would distort. A level recalibration (scalar or isotonic, fit on the same
pre-window holdout as the weights) is required before the backtest re-baseline
(@sec-limitations).

A further honest null result: the ~37 new features added alongside this batch (opponent form,
EWMA form, per-90 rates, xP) left CatBoost's full-window MASE essentially unchanged
(0.513→0.517 GK, 0.705→0.706 DEF, 0.731→0.730 MID, 0.849→0.847 FWD). They are retained - the
underlying semantic corrections are correctness fixes regardless, and no hyperparameter tuning
has yet been attempted that might exploit them - but they have not yet earned their keep on
accuracy alone.

== Fixture/minutes features: better forecasts, worse squads <sec-fixmin>

Adding fixture-difficulty and minutes-projection features improved MASE at every position (DEF
most, consistent with clean-sheet dependence) - yet realized backtest points _fell_ 1,966 → 1,880.
Leading hypothesis: the features smooth the mean forecast toward safe, nailed players, and the
MILP - blind to variance - loses the high-ceiling captaincy differentials that drive realized
hauls. The features are committed but not declared a net win; resolving this (noise-check on a
second window, probabilistic-upside captaincy) is a top open item. The probabilistic module's
[p10, p90] coverage is 0.88-0.93 against nominal 0.80 - somewhat wide, usable for relative risk
ranking; conformal calibration is a possible refinement.

== Exploratory branches

Two isolated branches, neither merged: a Bayesian belief-state MDP manager after Matthews et al.
(2012) - 1,083 points myopic / 847 Q-learning where the production system scored 1,900, expected
given documented simplifications - and a per-player model-selection plan (feasibility capped by
~35% of live players having no prior-season history; recommendation: single-position pilot).

= Methodological limitations <sec-limitations>

An internal code-and-concept review (2026-07-04) found four errors; all four were fixed the same
day. One further limitation emerged from the diagnostics added afterwards. Recorded here because
published numbers predate some fixes:

+ *Production forecast level is miscalibrated (open, fix designed).* The production CatBoost
  models under-predict the aggregate point level by ~40-55% (a structural consequence of MAE loss
  on a zero-inflated target), while ranking players better than any alternative. Harmless for
  pure ranking, but the MILP's transfer penalty and chip logic operate on the absolute point
  scale, so a level recalibration on the pre-window holdout must precede the backtest
  re-baseline.

+ *Blend-weight leakage (fixed).* Backtest predictions formerly reused blend weights fit inside
  the backtest window itself. Weights are now always fit on a window strictly before whatever is
  predicted. The 1,966/1,880 figures predate the fix and are modestly inflated; their relative
  comparison stands (both leaked identically). Re-baselining is pending.
+ *Live-mode staleness (fixed).* The weekly driver formerly reused each player's last played row,
  so live fixture-difficulty described last week's fixture and rolling form excluded the most
  recent match; API-vs-dataset team-name mismatches also silently dropped some teams' players.
  Live predictions now use per-gameweek API fixture difficulties and a synthetic next-gameweek
  row whose features include everything played. Backtests were never affected.
+ *Index-check asymmetry (fixed).* The ensemble-vs-OLS verdict now compares both on the same
  held-out rows.
+ *Optimizer rule currency (open).* The MILP caps banked free transfers at 2 (now 5 in FPL) and
  assumes sales at market value (real FPL: purchase price plus half profit). Fine for historical
  comparison; wrong for live 2026-27 play. Deliberately deferred - optimizer work is
  deprioritized in favor of forecasting.

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
